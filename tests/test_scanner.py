import asyncio
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chesscv import (
    BoardTransform,
    CameraOpenError,
    ChessBoardScanner,
    ModelOutputError,
    Occupant,
)
from chesscv.models import _ModelBoard
from chesscv.scanner import MODEL, SYSTEM_PROMPT, _CameraWorker, _Frame


def parsed_start_position() -> _ModelBoard:
    return _ModelBoard.model_validate(
        {
            "schema_version": "1",
            "status": "complete",
            "confidence": {
                "board_detection": "high",
                "grid_mapping": "high",
                "piece_recognition": "high",
            },
            "rows": {
                "row_1": "rnbqkbnr",
                "row_2": "pppppppp",
                "row_3": "........",
                "row_4": "........",
                "row_5": "........",
                "row_6": "........",
                "row_7": "PPPPPPPP",
                "row_8": "RNBQKBNR",
            },
        }
    )


class FakeResponses:
    def __init__(self, parsed: _ModelBoard | None, delay: float = 0.0) -> None:
        self.parsed = parsed
        self.delay = delay
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()

    async def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return SimpleNamespace(output_parsed=self.parsed)
        finally:
            self.active -= 1


class FakeClient:
    def __init__(self, responses: FakeResponses, close_delay: float = 0.0) -> None:
        self.responses = responses
        self.close_delay = close_delay
        self.closed = False
        self.close_started = asyncio.Event()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        self.closed = True


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.thread_ids: set[int] = {threading.get_ident()}
        self.frame = np.zeros((32, 48, 3), dtype=np.uint8)

    def _record(self) -> None:
        self.thread_ids.add(threading.get_ident())

    def isOpened(self) -> bool:
        self._record()
        return self.opened

    def set(self, prop: int, value: int) -> bool:
        self._record()
        return True

    def grab(self) -> bool:
        self._record()
        time.sleep(0.002)
        return not self.released

    def retrieve(self) -> tuple[bool, Any]:
        self._record()
        return True, self.frame

    def release(self) -> None:
        self._record()
        self.released = True


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    delay: float = 0.0,
    no_parsed_output: bool = False,
    close_delay: float = 0.0,
) -> tuple[Any, Any]:
    parsed = None if no_parsed_output else parsed_start_position()
    responses = FakeResponses(parsed, delay=delay)
    client = FakeClient(responses, close_delay=close_delay)
    captures: list[FakeCapture] = []

    def client_factory(**kwargs: Any) -> FakeClient:
        return client

    def capture_factory(source: int | str, backend: int) -> FakeCapture:
        capture = FakeCapture()
        captures.append(capture)
        return capture

    monkeypatch.setattr("chesscv.scanner.AsyncOpenAI", client_factory)
    monkeypatch.setattr("chesscv.scanner.cv2.VideoCapture", capture_factory)
    return SimpleNamespace(client=client, responses=responses), captures


@pytest.mark.asyncio
async def test_scanner_runs_inside_host_loop_and_delivers_validated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captures = install_fakes(monkeypatch)
    host_thread = threading.get_ident()
    scanner = ChessBoardScanner(
        api_key="host-key",
        interval=0.02,
        transform=BoardTransform.WHITE_AT_BOTTOM,
        warmup_frames=0,
    )
    subscription = scanner.subscribe()

    async with scanner:
        observation = await asyncio.wait_for(anext(subscription), timeout=2)

    call = fake.responses.calls[0]
    image = call["input"][0]["content"][1]
    assert observation.squares.piece_at("a8") is Occupant.BLACK_ROOK
    assert observation.squares.piece_at("e1") is Occupant.WHITE_KING
    assert call["model"] == MODEL == "gpt-5.6-luna"
    assert call["instructions"] == SYSTEM_PROMPT
    assert call["text_format"] is _ModelBoard
    assert call["reasoning"] == {"effort": "none"}
    assert call["store"] is False
    assert image["image_url"].startswith("data:image/jpeg;base64,")
    assert image["detail"] == "high"
    assert fake.responses.max_active == 1
    assert fake.client.closed
    assert captures[0].released
    assert len(captures[0].thread_ids) == 1
    assert host_thread not in captures[0].thread_ids


@pytest.mark.asyncio
async def test_latest_frame_and_latest_observation_replace_stale_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = install_fakes(monkeypatch)
    scanner = ChessBoardScanner(api_key="host-key")
    subscription = scanner.subscribe(buffer_size=1)
    captured_at = datetime.now(UTC)
    first = _Frame(sequence=1, captured_at=captured_at, jpeg=b"first")
    second = _Frame(sequence=2, captured_at=captured_at, jpeg=b"second")

    scanner._offer_frame(first)
    scanner._offer_frame(second)
    assert scanner._frames.get_nowait() is second

    observation_one = await scanner._analyze(first)
    observation_two = await scanner._analyze(second)
    scanner._publish(observation_one)
    scanner._publish(observation_two)

    assert await asyncio.wait_for(anext(subscription), timeout=1) is observation_two
    assert len(fake.responses.calls) == 2
    await scanner.aclose()


@pytest.mark.asyncio
async def test_no_subscribers_means_no_api_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _ = install_fakes(monkeypatch)
    scanner = ChessBoardScanner(api_key="host-key", interval=0.01, warmup_frames=0)

    async with scanner:
        await asyncio.sleep(0.06)

    assert fake.responses.calls == []


@pytest.mark.asyncio
async def test_close_cancels_in_flight_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake, _ = install_fakes(monkeypatch, delay=30)
    scanner = ChessBoardScanner(api_key="host-key", interval=0.01, warmup_frames=0)
    scanner.subscribe()
    await scanner.start()
    await asyncio.wait_for(fake.responses.started.wait(), timeout=2)

    await scanner.aclose()

    assert fake.responses.active == 0
    assert fake.client.closed


@pytest.mark.asyncio
async def test_camera_open_failure_closes_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = FakeResponses(parsed_start_position())
    client = FakeClient(responses)

    monkeypatch.setattr("chesscv.scanner.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "chesscv.scanner.cv2.VideoCapture",
        lambda source, backend: FakeCapture(opened=False),
    )
    scanner = ChessBoardScanner(api_key="host-key", warmup_frames=0)

    with pytest.raises(CameraOpenError):
        async with scanner:
            pass

    assert client.closed


@pytest.mark.asyncio
async def test_subscription_close_wakes_pending_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    scanner = ChessBoardScanner(api_key="host-key")
    subscription = scanner.subscribe()
    pending = asyncio.create_task(anext(subscription))
    await asyncio.sleep(0)

    subscription.close()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, timeout=1)
    await scanner.aclose()


@pytest.mark.asyncio
async def test_concurrent_lifecycle_owns_one_camera_and_one_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captures = install_fakes(monkeypatch)
    scanner = ChessBoardScanner(api_key="host-key", interval=0.01, warmup_frames=0)

    await asyncio.gather(scanner.start(), scanner.start())
    await asyncio.gather(scanner.aclose(), scanner.aclose())

    assert len(captures) == 1
    assert captures[0].released
    assert fake.client.close_calls == 1
    assert fake.client.closed
    assert not scanner.running


@pytest.mark.asyncio
async def test_cancelled_close_keeps_shared_cleanup_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, captures = install_fakes(monkeypatch, close_delay=0.1)
    scanner = ChessBoardScanner(api_key="host-key", interval=0.01, warmup_frames=0)
    await scanner.start()
    caller = asyncio.create_task(scanner.aclose())
    await asyncio.wait_for(fake.client.close_started.wait(), timeout=2)

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await scanner.aclose()

    assert fake.client.closed
    assert fake.client.close_calls == 1
    assert captures[0].released
    assert not scanner.running


@pytest.mark.asyncio
async def test_invalid_model_output_reaches_subscriber_after_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake, _ = install_fakes(monkeypatch, no_parsed_output=True)
    scanner = ChessBoardScanner(
        api_key="host-key",
        interval=0.01,
        warmup_frames=0,
        max_model_failures=2,
    )
    subscription = scanner.subscribe()

    await scanner.start()
    with pytest.raises(ModelOutputError, match="failed validation"):
        await asyncio.wait_for(anext(subscription), timeout=3)
    await scanner.aclose()

    assert len(fake.responses.calls) == 2


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_the_same_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    scanner = ChessBoardScanner(api_key="host-key", interval=0.01, warmup_frames=0)
    first = scanner.subscribe()
    second = scanner.subscribe()

    async with scanner:
        first_result, second_result = await asyncio.gather(
            asyncio.wait_for(anext(first), timeout=2),
            asyncio.wait_for(anext(second), timeout=2),
        )

    assert first_result is second_result


@pytest.mark.asyncio
async def test_camera_thread_schedules_only_one_pending_frame_delivery() -> None:
    class DeferredLoop:
        def __init__(self) -> None:
            self.callbacks: list[tuple[Any, tuple[Any, ...]]] = []

        def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
            self.callbacks.append((callback, args))

    deferred_loop = DeferredLoop()
    delivered: list[_Frame] = []
    ready = asyncio.get_running_loop().create_future()
    worker = _CameraWorker(
        loop=deferred_loop,
        source=0,
        backend=None,
        interval=1.0,
        width=None,
        height=None,
        max_dimension=1280,
        jpeg_quality=82,
        warmup_frames=0,
        ready=ready,
        on_frame=delivered.append,
        on_failure=lambda error: None,
    )
    captured_at = datetime.now(UTC)
    first = _Frame(sequence=1, captured_at=captured_at, jpeg=b"first")
    second = _Frame(sequence=2, captured_at=captured_at, jpeg=b"second")

    worker._queue_frame(first)
    worker._queue_frame(second)

    assert len(deferred_loop.callbacks) == 1
    callback, args = deferred_loop.callbacks.pop()
    callback(*args)
    assert delivered == [second]


def test_retry_delay_is_bounded_for_any_failure_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fakes(monkeypatch)
    scanner = ChessBoardScanner(api_key="host-key")

    assert scanner._retry_delay(10**9) == 8.0
