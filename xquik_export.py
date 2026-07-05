from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TEXT_COLUMNS = (
    "text",
    "tweet",
    "full_text",
    "content",
    "body",
    "headline",
    "title",
    "message",
)
DATE_COLUMNS = ("created_at", "date", "timestamp", "published")
SOURCE_ID_COLUMNS = ("id", "tweet_id", "post_id", "source_id", "link")


def _match_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    by_key = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in by_key:
            return by_key[candidate]
    return None


def _clean(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[column].fillna("").astype(str).str.strip()


def normalize_xquik_export(frame: pd.DataFrame) -> pd.DataFrame:
    text_column = _match_column(frame.columns, TEXT_COLUMNS)
    if text_column is None:
        return pd.DataFrame(columns=["text", "published", "source_id"])

    normalized = pd.DataFrame(index=frame.index)
    normalized["text"] = _clean(frame, text_column)
    normalized["published"] = _clean(frame, _match_column(frame.columns, DATE_COLUMNS))
    normalized["source_id"] = _clean(frame, _match_column(frame.columns, SOURCE_ID_COLUMNS))
    normalized = normalized[normalized["text"] != ""]
    return normalized.reset_index(drop=True)


def combine_export_text(frame: pd.DataFrame) -> str:
    normalized = normalize_xquik_export(frame)
    return "\n\n".join(normalized["text"].tolist())
