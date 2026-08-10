from __future__ import annotations

import asyncio
import base64
import sys
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import cv2
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageParam,
    ResponseInputParam,
    ResponseInputTextParam,
)
from pydantic import ValidationError

from .models import BoardObservation, BoardSquares, BoardTransform, _ModelBoard

MODEL = "gpt-5.6-luna"

SYSTEM_PROMPT = """You are a deterministic visual transcriber for one physical chessboard camera frame. Return exactly one result matching the supplied structured schema.

Select one physical 8x8 chessboard. If several are visible, select the complete board occupying the largest image area; break a tie by selecting the board nearest the image center.

The board fields row_1 through row_8 are image-space grid rows ordered from the detected board's top edge to its bottom edge. Every row is exactly eight characters ordered from the detected board's left edge to its right edge. Do not assign algebraic coordinates or infer chessboard orientation. The host software maps this image-space grid to squares.

Use these symbols:
P N B R Q K = exact white pawn, knight, bishop, rook, queen, king
p n b r q k = exact black pawn, knight, bishop, rook, queen, king
. = confidently empty square
x = definitely occupied, but piece color or type is not visually justified
? = occupancy itself cannot be determined

Locate the four board corners in image space, account for perspective, divide the board into an 8x8 grid, and inspect all 64 cells independently from top to bottom and left to right. Use an exact piece symbol only when its cell, color, and piece type are visually supported. Use x or ? for blur, glare, hands, occlusion, cropping, ambiguous silhouettes, or uncertain colors. Never guess to make the position look plausible.

Use only this frame. Do not use previous frames, expected motion, material counts, starting-position assumptions, move legality, king counts, or symmetry to repair the observation. Transcribe visually impossible or illegal positions exactly as seen.

Do not determine side to move, castling rights, en-passant state, clocks, check status, legal moves, or full FEN because one image does not establish them.

Status rules:
complete = the board and grid are located and every square is an exact piece or confidently empty
partial = the board and grid are located and at least one square is x or ?
not_found = no qualifying 8x8 chessboard is detected
unusable = a possible board exists but the frame cannot support a reliable grid transcription

For complete, board_detection, grid_mapping, and piece_recognition must each be high or medium. For partial, board_detection and grid_mapping must be high or medium, and piece_recognition must not be none. For not_found, return ???????? for all eight rows and set all three confidences to none. For unusable, return ???????? for all eight rows, set piece_recognition to none, and set board_detection to high, medium, or low. Confidence is categorical: high means direct and clear evidence, medium means adequate evidence with minor degradation, low means tentative evidence, and none means no supporting evidence."""


class ScannerError(RuntimeError):
    pass


class ScannerClosedError(ScannerError):
    pass


class CameraOpenError(ScannerError):
    pass


class CameraReadError(ScannerError):
    pass


class CameraShutdownError(ScannerError):
    pass


class ModelOutputError(ScannerError):
    pass


@dataclass(frozen=True, slots=True)
class _Frame:
    sequence: int
    captured_at: datetime
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class _Failure:
    error: BaseException


_END = object()


class BoardSubscription(AsyncIterator[BoardObservation]):
    def __init__(self, scanner: ChessBoardScanner, buffer_size: int) -> None:
        self._scanner = scanner
        self._queue: asyncio.Queue[BoardObservation | _Failure | object] = asyncio.Queue(
            maxsize=buffer_size
        )
        self._closed = False

    def __aiter__(self) -> BoardSubscription:
        return self

    async def __anext__(self) -> BoardObservation:
        if self._closed:
            raise StopAsyncIteration
        try:
            item = await self._queue.get()
        except asyncio.CancelledError:
            self.close()
            raise
        if item is _END:
            self.close()
            raise StopAsyncIteration
        if isinstance(item, _Failure):
            self.close()
            raise item.error
        if not isinstance(item, BoardObservation):
            self.close()
            raise ScannerError("invalid subscription item")
        return item

    async def __aenter__(self) -> BoardSubscription:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _offer(self, item: BoardObservation | _Failure | object) -> None:
        if self._closed:
            return
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(item)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._scanner._unsubscribe(self)
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(_END)

    async def aclose(self) -> None:
        self.close()


class _CameraWorker:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        source: int | str,
        backend: int | None,
        interval: float,
        width: int | None,
        height: int | None,
        max_dimension: int,
        jpeg_quality: int,
        warmup_frames: int,
        ready: asyncio.Future[None],
        on_frame: Any,
        on_failure: Any,
    ) -> None:
        self._loop = loop
        self._source = source
        self._backend = backend
        self._interval = interval
        self._width = width
        self._height = height
        self._max_dimension = max_dimension
        self._jpeg_quality = jpeg_quality
        self._warmup_frames = warmup_frames
        self._ready = ready
        self._on_frame = on_frame
        self._on_failure = on_failure
        self._stop = threading.Event()
        self._delivery_lock = threading.Lock()
        self._pending_frame: _Frame | None = None
        self._delivery_scheduled = False
        self._thread = threading.Thread(
            target=self._run,
            name="chesscv-camera",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _post(self, callback: Any, *args: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            return

    def _queue_frame(self, frame: _Frame) -> None:
        with self._delivery_lock:
            self._pending_frame = frame
            if self._delivery_scheduled:
                return
            self._delivery_scheduled = True
        self._post(self._deliver_frame)

    def _deliver_frame(self) -> None:
        with self._delivery_lock:
            frame = self._pending_frame
            self._pending_frame = None
            self._delivery_scheduled = False
        if frame is not None:
            self._on_frame(frame)

    def _open_capture(self) -> Any:
        if self._backend is not None:
            backends = (self._backend,)
        elif sys.platform == "win32":
            backends = tuple(dict.fromkeys((cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)))
        else:
            backends = (cv2.CAP_ANY,)
        for backend in backends:
            capture = cv2.VideoCapture(self._source, backend)
            if capture.isOpened():
                return capture
            capture.release()
        raise CameraOpenError(f"could not open camera {self._source!r}")

    def _configure_capture(self, capture: Any) -> None:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self._width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

    def _encode(self, frame: Any) -> bytes:
        height, width = frame.shape[:2]
        largest = max(height, width)
        if largest > self._max_dimension:
            scale = self._max_dimension / largest
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality),
        )
        if not success:
            raise CameraReadError("could not encode a camera frame")
        return encoded.tobytes()

    def _run(self) -> None:
        capture = None
        ready_sent = False
        try:
            capture = self._open_capture()
            self._configure_capture(capture)
            for _ in range(self._warmup_frames):
                if self._stop.is_set():
                    break
                if not capture.grab():
                    raise CameraReadError("camera failed during warmup")
            sequence = 0
            next_sample = time.monotonic()
            failures = 0
            while not self._stop.is_set():
                if not capture.grab():
                    failures += 1
                    if failures >= 5:
                        raise CameraReadError("camera failed to capture five consecutive frames")
                    self._stop.wait(0.05)
                    continue
                failures = 0
                now = time.monotonic()
                if now < next_sample:
                    continue
                success, frame = capture.retrieve()
                if not success or frame is None:
                    raise CameraReadError("camera failed to retrieve a frame")
                sequence += 1
                jpeg = self._encode(frame)
                captured = _Frame(
                    sequence=sequence,
                    captured_at=datetime.now(UTC),
                    jpeg=jpeg,
                )
                self._queue_frame(captured)
                if not ready_sent:
                    self._post(self._ready.set_result, None)
                    ready_sent = True
                next_sample += self._interval
                if next_sample <= now:
                    next_sample = now + self._interval
        except BaseException as error:
            if not ready_sent:
                self._post(self._ready.set_exception, error)
            elif not self._stop.is_set():
                self._post(self._on_failure, error)
        finally:
            if capture is not None:
                capture.release()


class ChessBoardScanner:
    def __init__(
        self,
        *,
        api_key: str,
        camera: int | str = 0,
        interval: float = 1.0,
        transform: BoardTransform | str = BoardTransform.WHITE_AT_BOTTOM,
        frame_width: int | None = 1280,
        frame_height: int | None = 720,
        max_image_dimension: int = 1280,
        jpeg_quality: int = 82,
        image_detail: Literal["low", "high", "auto", "original"] = "high",
        request_timeout: float = 20.0,
        camera_backend: int | None = None,
        warmup_frames: int = 3,
        camera_start_timeout: float = 5.0,
        camera_shutdown_timeout: float = 3.0,
        max_model_failures: int = 3,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key cannot be empty")
        if interval <= 0:
            raise ValueError("interval must be greater than zero")
        if frame_width is not None and frame_width <= 0:
            raise ValueError("frame_width must be greater than zero")
        if frame_height is not None and frame_height <= 0:
            raise ValueError("frame_height must be greater than zero")
        if max_image_dimension < 256:
            raise ValueError("max_image_dimension must be at least 256")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        if warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")
        if camera_start_timeout <= 0 or camera_shutdown_timeout <= 0:
            raise ValueError("camera timeouts must be greater than zero")
        if max_model_failures < 1:
            raise ValueError("max_model_failures must be at least one")
        self._camera_source = camera
        self._interval = interval
        self._transform = BoardTransform(transform)
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._max_image_dimension = max_image_dimension
        self._jpeg_quality = jpeg_quality
        self._image_detail: Literal["low", "high", "auto", "original"] = image_detail
        self._camera_backend = camera_backend
        self._warmup_frames = warmup_frames
        self._camera_start_timeout = camera_start_timeout
        self._camera_shutdown_timeout = camera_shutdown_timeout
        self._max_model_failures = max_model_failures
        self._client = AsyncOpenAI(api_key=api_key, timeout=request_timeout, max_retries=1)
        self._frames: asyncio.Queue[_Frame] = asyncio.Queue(maxsize=1)
        self._subscriptions: set[BoardSubscription] = set()
        self._camera: _CameraWorker | None = None
        self._analysis_task: asyncio.Task[None] | None = None
        self._running = False
        self._closing = False
        self._closed = False
        self._failure: BaseException | None = None
        self._last_error: BaseException | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    @property
    def transform(self) -> BoardTransform:
        return self._transform

    async def __aenter__(self) -> ChessBoardScanner:
        try:
            await self.start()
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._closed or self._closing or self._close_task is not None:
                raise ScannerClosedError("the scanner is closed")
            if self._failure is not None:
                raise ScannerError("a failed scanner cannot be restarted") from self._failure
            if self._running:
                return
            loop = asyncio.get_running_loop()
            ready: asyncio.Future[None] = loop.create_future()
            camera = _CameraWorker(
                loop=loop,
                source=self._camera_source,
                backend=self._camera_backend,
                interval=self._interval,
                width=self._frame_width,
                height=self._frame_height,
                max_dimension=self._max_image_dimension,
                jpeg_quality=self._jpeg_quality,
                warmup_frames=self._warmup_frames,
                ready=ready,
                on_frame=self._offer_frame,
                on_failure=self._fail,
            )
            self._camera = camera
            camera.start()
            try:
                await asyncio.wait_for(asyncio.shield(ready), self._camera_start_timeout)
            except BaseException as error:
                camera.stop()
                stopped = await asyncio.to_thread(
                    camera.join,
                    self._camera_shutdown_timeout,
                )
                if stopped:
                    self._camera = None
                else:
                    raise CameraShutdownError("camera thread did not stop in time") from error
                raise
            if self._failure is not None:
                error = self._failure
                camera.stop()
                stopped = await asyncio.to_thread(
                    camera.join,
                    self._camera_shutdown_timeout,
                )
                if stopped:
                    self._camera = None
                raise error
            if self._close_task is not None or self._closing or self._closed:
                camera.stop()
                stopped = await asyncio.to_thread(
                    camera.join,
                    self._camera_shutdown_timeout,
                )
                if stopped:
                    self._camera = None
                raise ScannerClosedError("the scanner was closed during startup")
            self._running = True
            self._analysis_task = asyncio.create_task(
                self._analysis_loop(),
                name="chesscv-analysis",
            )
            self._analysis_task.add_done_callback(self._analysis_done)

    def subscribe(self, *, buffer_size: int = 1) -> BoardSubscription:
        if self._closed or self._closing or self._close_task is not None:
            raise ScannerClosedError("the scanner is closed")
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least one")
        subscription = BoardSubscription(self, buffer_size)
        self._subscriptions.add(subscription)
        if self._failure is not None:
            subscription._offer(_Failure(self._failure))
        return subscription

    async def aclose(self) -> None:
        task = self._close_task
        if task is None:
            if self._closed:
                return
            task = asyncio.create_task(self._close(), name="chesscv-close")
            self._close_task = task
        await asyncio.shield(task)

    async def _close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True
            self._running = False
            cleanup_error: BaseException | None = None
            try:
                task = self._analysis_task
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                self._analysis_task = None
                camera = self._camera
                if camera is not None:
                    camera.stop()
                    stopped = await asyncio.to_thread(
                        camera.join,
                        self._camera_shutdown_timeout,
                    )
                    if not stopped:
                        cleanup_error = CameraShutdownError("camera thread did not stop in time")
                self._camera = None
            except BaseException as error:
                cleanup_error = error
            try:
                await self._client.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
            finally:
                self._finish_subscriptions(_END)
                self._closed = True
                self._closing = False
            if cleanup_error is not None:
                raise cleanup_error

    def _unsubscribe(self, subscription: BoardSubscription) -> None:
        self._subscriptions.discard(subscription)

    def _offer_frame(self, frame: _Frame) -> None:
        if self._closing or self._closed or self._failure is not None:
            return
        if self._frames.full():
            self._frames.get_nowait()
        self._frames.put_nowait(frame)

    def _publish(self, observation: BoardObservation) -> None:
        for subscription in tuple(self._subscriptions):
            subscription._offer(observation)

    def _finish_subscriptions(self, item: _Failure | object) -> None:
        for subscription in tuple(self._subscriptions):
            subscription._offer(item)
        self._subscriptions.clear()

    def _fail(self, error: BaseException) -> None:
        if self._closing or self._closed or self._failure is not None:
            return
        self._failure = error
        self._last_error = error
        self._running = False
        if self._camera is not None:
            self._camera.stop()
        task = self._analysis_task
        if task is not None and not task.done():
            task.cancel()
        self._finish_subscriptions(_Failure(error))

    def _analysis_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or self._closing or self._closed:
            return
        error = task.exception()
        if error is not None:
            self._fail(error)

    def _retryable(self, error: OpenAIError) -> bool:
        if isinstance(error, APIConnectionError):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        return False

    async def _wait_or_close(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise

    def _retry_delay(self, failures: int) -> float:
        exponent = min(max(failures - 1, 0), 4)
        return min(8.0, 0.5 * 2**exponent)

    async def _analysis_loop(self) -> None:
        api_failures = 0
        model_failures = 0
        while True:
            frame = await self._frames.get()
            if not self._subscriptions:
                api_failures = 0
                model_failures = 0
                continue
            try:
                observation = await self._analyze(frame)
            except asyncio.CancelledError:
                raise
            except (ModelOutputError, ValidationError) as error:
                self._last_error = error
                api_failures = 0
                model_failures += 1
                if model_failures >= self._max_model_failures:
                    raise ModelOutputError(
                        f"the model failed validation {model_failures} consecutive times"
                    ) from error
                await self._wait_or_close(self._retry_delay(model_failures))
                continue
            except OpenAIError as error:
                if not self._retryable(error):
                    raise
                self._last_error = error
                api_failures += 1
                await self._wait_or_close(self._retry_delay(api_failures))
                continue
            self._last_error = None
            api_failures = 0
            model_failures = 0
            self._publish(observation)

    async def _analyze(self, frame: _Frame) -> BoardObservation:
        encoded = base64.b64encode(frame.jpeg).decode("ascii")
        image_url = f"data:image/jpeg;base64,{encoded}"
        started = time.perf_counter()
        text_input: ResponseInputTextParam = {
            "type": "input_text",
            "text": "Transcribe the physical chessboard in this webcam frame.",
        }
        image_input: ResponseInputImageParam = {
            "type": "input_image",
            "image_url": image_url,
            "detail": self._image_detail,
        }
        message: EasyInputMessageParam = {
            "role": "user",
            "content": [text_input, image_input],
        }
        request_input: ResponseInputParam = [message]
        response = await self._client.responses.parse(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=request_input,
            text_format=_ModelBoard,
            reasoning={"effort": "none"},
            max_output_tokens=768,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            status = getattr(response, "status", None)
            incomplete = getattr(response, "incomplete_details", None)
            refusal = any(
                getattr(content, "type", None) == "refusal"
                for output in getattr(response, "output", ())
                for content in getattr(output, "content", ())
            )
            reason = "refusal" if refusal else "missing parsed output"
            if status is not None:
                reason = f"{reason}; status={status}"
            if incomplete is not None:
                reason = f"{reason}; incomplete_details={incomplete}"
            raise ModelOutputError(f"model response was unusable: {reason}")
        return BoardObservation(
            schema_version=parsed.schema_version,
            sequence=frame.sequence,
            captured_at=frame.captured_at,
            analysis_ms=max(0, round((time.perf_counter() - started) * 1000)),
            transform=self._transform,
            status=parsed.status,
            confidence=parsed.confidence,
            squares=BoardSquares.from_image_rows(parsed.rows, self._transform),
        )
