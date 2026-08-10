# chesscv

`chesscv` is an asyncio-native library that samples a USB webcam, sends the newest frame to `gpt-5.6-luna`, validates the Structured Output with Pydantic, and publishes a stable 64-square map as both a model and `orjson` bytes.

It owns neither the process nor the event loop. The host passes the API key explicitly and controls startup and shutdown.

## Add to a uv host project

From the host software's directory, add this checkout as a local dependency:

```powershell
uv add --editable ..\chesscv
```

Use the package's registry name instead of the path after publishing it.

For development inside this checkout:

```powershell
uv sync
```

## Host integration

```python
from chesscv import BoardTransform, ChessBoardScanner


async def consume_chessboard(api_key: str) -> None:
    scanner = ChessBoardScanner(
        api_key=api_key,
        camera=0,
        interval=1.0,
        transform=BoardTransform.WHITE_AT_BOTTOM,
    )
    async with scanner:
        async with scanner.subscribe() as observations:
            async for observation in observations:
                validated_model = observation
                json_bytes = observation.to_orjson()
                occupancy_only = observation.occupancy_orjson()
                e4 = observation.squares.piece_at("e4")
```

`subscribe()` is a bounded latest-value async iterator. Multiple subscribers are supported. A slow subscriber loses stale observations instead of blocking capture or consuming unbounded memory.

The camera thread continuously drains the webcam but retrieves and JPEG-encodes only at `interval` boundaries. There is one persistent async OpenAI client and at most one inference request in flight. When inference is slower than capture, pending frames are replaced by the newest frame.

## Square contract

`observation.squares` always contains all 64 keys in canonical order `a8` through `h1`. Values are:

| Value | Meaning |
|---|---|
| `P N B R Q K` | White piece |
| `p n b r q k` | Black piece |
| `.` | Confidently empty |
| `x` | Occupied, piece identity uncertain |
| `?` | Occupancy uncertain |

The scanner does not infer moves, legality, side to move, castling rights, en passant, or clocks. Diff `occupancy_orjson()` or the square model in the host. A sensible physical-board policy is to accept a change only after two matching `complete` observations.

## Camera transform

The transform is authoritative and fixed for the mounted camera. Its value names the algebraic square at the image-space top-left, top-right, bottom-left, and bottom-right board corners. Luna transcribes only top-to-bottom, left-to-right image cells; the library applies the transform locally, so square orientation does not depend on model inference.

Common transforms are:

| Mount | Transform |
|---|---|
| White side at image bottom | `BoardTransform.WHITE_AT_BOTTOM` |
| Black side at image bottom | `BoardTransform.BLACK_AT_BOTTOM` |

All eight rotations and mirrorings are available on `BoardTransform`. Raw OpenCV capture is normally not mirrored, even when webcam preview software displays a mirror image.

## Lifecycle and failures

`start()` and `aclose()` are idempotent, and the scanner is intended to be used as an async context manager. It never calls `asyncio.run()`.

Camera-open failures are raised by `start()` or `__aenter__()`. Fatal camera and API failures are raised by active subscriptions. Transient connection, timeout, rate-limit, and server failures are retried against newer frames with bounded backoff and are visible through `scanner.last_error`. Refused or invalid model outputs become a terminal `ModelOutputError` after three consecutive failures by default.
