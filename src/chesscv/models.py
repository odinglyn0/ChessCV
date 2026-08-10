from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

import orjson
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


class Occupant(StrEnum):
    WHITE_PAWN = "P"
    WHITE_KNIGHT = "N"
    WHITE_BISHOP = "B"
    WHITE_ROOK = "R"
    WHITE_QUEEN = "Q"
    WHITE_KING = "K"
    BLACK_PAWN = "p"
    BLACK_KNIGHT = "n"
    BLACK_BISHOP = "b"
    BLACK_ROOK = "r"
    BLACK_QUEEN = "q"
    BLACK_KING = "k"
    EMPTY = "."
    OCCUPIED_UNKNOWN = "x"
    UNKNOWN = "?"


class Square(StrEnum):
    A8 = "a8"
    B8 = "b8"
    C8 = "c8"
    D8 = "d8"
    E8 = "e8"
    F8 = "f8"
    G8 = "g8"
    H8 = "h8"
    A7 = "a7"
    B7 = "b7"
    C7 = "c7"
    D7 = "d7"
    E7 = "e7"
    F7 = "f7"
    G7 = "g7"
    H7 = "h7"
    A6 = "a6"
    B6 = "b6"
    C6 = "c6"
    D6 = "d6"
    E6 = "e6"
    F6 = "f6"
    G6 = "g6"
    H6 = "h6"
    A5 = "a5"
    B5 = "b5"
    C5 = "c5"
    D5 = "d5"
    E5 = "e5"
    F5 = "f5"
    G5 = "g5"
    H5 = "h5"
    A4 = "a4"
    B4 = "b4"
    C4 = "c4"
    D4 = "d4"
    E4 = "e4"
    F4 = "f4"
    G4 = "g4"
    H4 = "h4"
    A3 = "a3"
    B3 = "b3"
    C3 = "c3"
    D3 = "d3"
    E3 = "e3"
    F3 = "f3"
    G3 = "g3"
    H3 = "h3"
    A2 = "a2"
    B2 = "b2"
    C2 = "c2"
    D2 = "d2"
    E2 = "e2"
    F2 = "f2"
    G2 = "g2"
    H2 = "h2"
    A1 = "a1"
    B1 = "b1"
    C1 = "c1"
    D1 = "d1"
    E1 = "e1"
    F1 = "f1"
    G1 = "g1"
    H1 = "h1"


SQUARE_ORDER = tuple(Square)


class BoardTransform(StrEnum):
    A8_H8_A1_H1 = "a8_h8_a1_h1"
    A1_A8_H1_H8 = "a1_a8_h1_h8"
    H1_A1_H8_A8 = "h1_a1_h8_a8"
    H8_H1_A8_A1 = "h8_h1_a8_a1"
    H8_A8_H1_A1 = "h8_a8_h1_a1"
    H1_H8_A1_A8 = "h1_h8_a1_a8"
    A1_H1_A8_H8 = "a1_h1_a8_h8"
    A8_A1_H8_H1 = "a8_a1_h8_h1"
    WHITE_AT_BOTTOM = "a8_h8_a1_h1"
    BLACK_AT_BOTTOM = "h1_a1_h8_a8"

    @property
    def corners(self) -> tuple[Square, Square, Square, Square]:
        top_left, top_right, bottom_left, bottom_right = self.value.split("_")
        return (
            Square(top_left),
            Square(top_right),
            Square(bottom_left),
            Square(bottom_right),
        )


class DetectionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    UNUSABLE = "unusable"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class BoardConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    board_detection: ConfidenceLevel
    grid_mapping: ConfidenceLevel
    piece_recognition: ConfidenceLevel


def _validate_semantics(
    status: DetectionStatus,
    confidence: BoardConfidence,
    encoded: str,
) -> None:
    unresolved = "?" in encoded or "x" in encoded
    if status is DetectionStatus.COMPLETE:
        if unresolved:
            raise ValueError("complete observations cannot contain unresolved squares")
        if confidence.board_detection in {ConfidenceLevel.LOW, ConfidenceLevel.NONE}:
            raise ValueError("complete observations require reliable board detection")
        if confidence.grid_mapping in {ConfidenceLevel.LOW, ConfidenceLevel.NONE}:
            raise ValueError("complete observations require a reliable grid mapping")
        if confidence.piece_recognition in {ConfidenceLevel.LOW, ConfidenceLevel.NONE}:
            raise ValueError("complete observations require reliable piece recognition")
    elif status is DetectionStatus.PARTIAL:
        if not unresolved:
            raise ValueError("partial observations must contain unresolved squares")
        if confidence.board_detection in {ConfidenceLevel.LOW, ConfidenceLevel.NONE}:
            raise ValueError("partial observations require reliable board detection")
        if confidence.grid_mapping in {ConfidenceLevel.LOW, ConfidenceLevel.NONE}:
            raise ValueError("partial observations require a reliable grid mapping")
        if confidence.piece_recognition is ConfidenceLevel.NONE:
            raise ValueError("partial observations require some piece recognition")
    else:
        if encoded != "?" * 64:
            raise ValueError("non-position observations must mark every square unknown")
        if confidence.piece_recognition is not ConfidenceLevel.NONE:
            raise ValueError("non-position observations cannot claim piece recognition")
    if status is DetectionStatus.NOT_FOUND:
        if confidence.board_detection is not ConfidenceLevel.NONE:
            raise ValueError("not_found cannot claim board detection confidence")
        if confidence.grid_mapping is not ConfidenceLevel.NONE:
            raise ValueError("not_found cannot claim grid mapping confidence")
    elif confidence.board_detection is ConfidenceLevel.NONE:
        raise ValueError("detected observations require board detection confidence")


class BoardSquares(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    a8: Occupant
    b8: Occupant
    c8: Occupant
    d8: Occupant
    e8: Occupant
    f8: Occupant
    g8: Occupant
    h8: Occupant
    a7: Occupant
    b7: Occupant
    c7: Occupant
    d7: Occupant
    e7: Occupant
    f7: Occupant
    g7: Occupant
    h7: Occupant
    a6: Occupant
    b6: Occupant
    c6: Occupant
    d6: Occupant
    e6: Occupant
    f6: Occupant
    g6: Occupant
    h6: Occupant
    a5: Occupant
    b5: Occupant
    c5: Occupant
    d5: Occupant
    e5: Occupant
    f5: Occupant
    g5: Occupant
    h5: Occupant
    a4: Occupant
    b4: Occupant
    c4: Occupant
    d4: Occupant
    e4: Occupant
    f4: Occupant
    g4: Occupant
    h4: Occupant
    a3: Occupant
    b3: Occupant
    c3: Occupant
    d3: Occupant
    e3: Occupant
    f3: Occupant
    g3: Occupant
    h3: Occupant
    a2: Occupant
    b2: Occupant
    c2: Occupant
    d2: Occupant
    e2: Occupant
    f2: Occupant
    g2: Occupant
    h2: Occupant
    a1: Occupant
    b1: Occupant
    c1: Occupant
    d1: Occupant
    e1: Occupant
    f1: Occupant
    g1: Occupant
    h1: Occupant

    @classmethod
    def from_image_rows(cls, rows: _ImageRows, transform: BoardTransform) -> Self:
        top_left, top_right, bottom_left, bottom_right = transform.corners
        files = "abcdefgh"
        top_left_file = files.index(top_left.value[0])
        top_right_file = files.index(top_right.value[0])
        bottom_left_file = files.index(bottom_left.value[0])
        bottom_right_file = files.index(bottom_right.value[0])
        top_left_rank = int(top_left.value[1])
        top_right_rank = int(top_right.value[1])
        bottom_left_rank = int(bottom_left.value[1])
        bottom_right_rank = int(bottom_right.value[1])
        file_column_step = (top_right_file - top_left_file) // 7
        file_row_step = (bottom_left_file - top_left_file) // 7
        rank_column_step = (top_right_rank - top_left_rank) // 7
        rank_row_step = (bottom_left_rank - top_left_rank) // 7
        expected_bottom_right_file = top_left_file + 7 * (file_column_step + file_row_step)
        expected_bottom_right_rank = top_left_rank + 7 * (rank_column_step + rank_row_step)
        if (
            expected_bottom_right_file != bottom_right_file
            or expected_bottom_right_rank != bottom_right_rank
        ):
            raise ValueError("invalid board transform")
        values: dict[str, Occupant] = {}
        for image_row, row in enumerate(rows.values()):
            for image_column, value in enumerate(row):
                file_index = (
                    top_left_file + image_column * file_column_step + image_row * file_row_step
                )
                rank = top_left_rank + image_column * rank_column_step + image_row * rank_row_step
                values[f"{files[file_index]}{rank}"] = Occupant(value)
        return cls.model_validate(values)

    def piece_at(self, square: Square | str) -> Occupant:
        return getattr(self, Square(square).value)

    def to_orjson(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json"))


class BoardObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    sequence: Annotated[int, Field(ge=1)]
    captured_at: datetime
    analysis_ms: Annotated[int, Field(ge=0)]
    transform: BoardTransform
    status: DetectionStatus
    confidence: BoardConfidence
    squares: BoardSquares

    @field_validator("captured_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        encoded = "".join(self.squares.piece_at(square).value for square in SQUARE_ORDER)
        _validate_semantics(self.status, self.confidence, encoded)
        return self

    def to_orjson(self) -> bytes:
        return orjson.dumps(self.model_dump(mode="json"))

    def occupancy_orjson(self) -> bytes:
        return self.squares.to_orjson()


RankString = Annotated[
    str,
    StringConstraints(strict=True, min_length=8, max_length=8, pattern=r"^[PNBRQKpnbrqk.x?]{8}$"),
]


class _ImageRows(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    row_1: RankString
    row_2: RankString
    row_3: RankString
    row_4: RankString
    row_5: RankString
    row_6: RankString
    row_7: RankString
    row_8: RankString

    def encoded(self) -> str:
        return "".join(self.values())

    def values(self) -> tuple[str, ...]:
        return (
            self.row_1,
            self.row_2,
            self.row_3,
            self.row_4,
            self.row_5,
            self.row_6,
            self.row_7,
            self.row_8,
        )


class _ModelBoard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    status: DetectionStatus
    confidence: BoardConfidence
    rows: _ImageRows

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        _validate_semantics(self.status, self.confidence, self.rows.encoded())
        return self
