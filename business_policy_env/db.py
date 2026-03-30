from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any


def _create_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS episode_actions (
            episode_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            valid INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    return connection


@dataclass
class ActionLogger:
    _connection: sqlite3.Connection = field(default_factory=_create_connection)
    _episode_id: str = field(init=False)

    def __post_init__(self) -> None:
        self._episode_id = self.new_episode()

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def new_episode(self) -> str:
        self._episode_id = uuid.uuid4().hex[:8]
        # Keep only the active episode in this in-memory logger.
        self._connection.execute("DELETE FROM episode_actions")
        self._connection.commit()
        return self._episode_id

    def log_action(
        self,
        *,
        step_index: int,
        action_type: str,
        payload: dict[str, Any],
        timestamp: str,
        valid: bool,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO episode_actions (episode_id, step_index, action_type, payload_json, timestamp, valid)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self._episode_id,
                step_index,
                action_type,
                json.dumps(payload, sort_keys=True),
                timestamp,
                int(valid),
            ),
        )
        self._connection.commit()

    def get_episode_actions(self) -> list[dict[str, Any]]:
        cursor = self._connection.execute(
            """
            SELECT step_index, action_type, payload_json, timestamp, valid
            FROM episode_actions
            WHERE episode_id = ?
            ORDER BY step_index ASC
            """,
            (self._episode_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "step_index": int(step_index),
                "action_type": str(action_type),
                "payload": json.loads(payload_json),
                "timestamp": str(timestamp),
                "valid": int(valid),
            }
            for step_index, action_type, payload_json, timestamp, valid in rows
        ]

    def total_logged(self) -> int:
        cursor = self._connection.execute(
            "SELECT COUNT(*) FROM episode_actions WHERE episode_id = ?",
            (self._episode_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def close(self) -> None:
        self._connection.close()
