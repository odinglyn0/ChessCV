from datetime import UTC, datetime

import orjson
import pytest
from pydantic import ValidationError

from chesscv import (
    BoardConfidence,
    BoardObservation,
    BoardSquares,
    BoardTransform,
    ConfidenceLevel,
    DetectionStatus,
    Occupant,
    Square,
)
from chesscv.models import _ImageRows, _ModelBoard


def square_values(default: Occupant = Occupant.EMPTY) -> dict[str, Occupant]:
    return {square.value: default for square in Square}


def test_observation_is_fixed_order_validated_and_serialized() -> None:
    values = square_values()
    values["e4"] = Occupant.WHITE_PAWN
    values["c6"] = Occupant.BLACK_KNIGHT
    squares = BoardSquares.model_validate(values)
    observation = BoardObservation(
        schema_version="1",
        sequence=7,
        captured_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
        analysis_ms=123,
        transform=BoardTransform.WHITE_AT_BOTTOM,
        status=DetectionStatus.COMPLETE,
        confidence=BoardConfidence(
            board_detection=ConfidenceLevel.HIGH,
            grid_mapping=ConfidenceLevel.HIGH,
            piece_recognition=ConfidenceLevel.HIGH,
        ),
        squares=squares,
    )

    first = observation.to_orjson()
    second = observation.to_orjson()
    decoded = orjson.loads(first)

    assert first == second
    assert list(decoded["squares"]) == [square.value for square in Square]
    assert decoded["squares"]["e4"] == "P"
    assert decoded["squares"]["c6"] == "n"
    assert squares.piece_at("e4") is Occupant.WHITE_PAWN
    assert squares.piece_at(Square.C6) is Occupant.BLACK_KNIGHT
    assert orjson.loads(observation.occupancy_orjson())["a8"] == "."
    copied = observation.model_copy(update={"sequence": 8})
    assert orjson.loads(copied.to_orjson())["sequence"] == 8


def test_observation_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        BoardObservation(
            schema_version="1",
            sequence=1,
            captured_at=datetime(2026, 8, 10),
            analysis_ms=1,
            transform=BoardTransform.WHITE_AT_BOTTOM,
            status=DetectionStatus.COMPLETE,
            confidence=BoardConfidence(
                board_detection=ConfidenceLevel.HIGH,
                grid_mapping=ConfidenceLevel.HIGH,
                piece_recognition=ConfidenceLevel.HIGH,
            ),
            squares=BoardSquares.model_validate(square_values()),
        )


def test_model_schema_rejects_inconsistent_status() -> None:
    data = {
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
            "row_5": "....?...",
            "row_6": "........",
            "row_7": "PPPPPPPP",
            "row_8": "RNBQKBNR",
        },
    }

    with pytest.raises(ValidationError, match="complete observations"):
        _ModelBoard.model_validate(data)


@pytest.mark.parametrize("transform", list(BoardTransform))
def test_all_camera_transforms_are_applied_locally(transform: BoardTransform) -> None:
    rows = _ImageRows(
        row_1="P......N",
        row_2="........",
        row_3="........",
        row_4="........",
        row_5="........",
        row_6="........",
        row_7="........",
        row_8="B......R",
    )
    squares = BoardSquares.from_image_rows(rows, transform)
    top_left, top_right, bottom_left, bottom_right = transform.corners

    assert squares.piece_at(top_left) is Occupant.WHITE_PAWN
    assert squares.piece_at(top_right) is Occupant.WHITE_KNIGHT
    assert squares.piece_at(bottom_left) is Occupant.WHITE_BISHOP
    assert squares.piece_at(bottom_right) is Occupant.WHITE_ROOK


def test_public_observation_rejects_inconsistent_status() -> None:
    with pytest.raises(ValidationError, match="non-position observations"):
        BoardObservation(
            schema_version="1",
            sequence=1,
            captured_at=datetime.now(UTC),
            analysis_ms=1,
            transform=BoardTransform.WHITE_AT_BOTTOM,
            status=DetectionStatus.NOT_FOUND,
            confidence=BoardConfidence(
                board_detection=ConfidenceLevel.NONE,
                grid_mapping=ConfidenceLevel.NONE,
                piece_recognition=ConfidenceLevel.NONE,
            ),
            squares=BoardSquares.model_validate(square_values()),
        )
