"""
Conversation history manager.

Maintains the last N interactions per user for context-aware responses.
"""

from collections import defaultdict, deque

import config


class HistoryManager:
    """Track per-user conversation history (last N exchanges)."""

    def __init__(self, max_per_user: int = config.MAX_HISTORY_PER_USER):
        self.max_per_user = max_per_user
        # user_id -> deque of {"role": ..., "content": ..., "type": ...}
        self._store: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max_per_user * 2)  # pairs of user + assistant
        )
        # Last interaction summary per user (for /summarize)
        self._last_interaction: dict[int, dict] = {}

    def add(self, user_id: int, role: str, content: str,
            interaction_type: str = "text") -> None:
        """Add a message to the user's history."""
        self._store[user_id].append({
            "role": role,
            "content": content,
        })
        # Track last interaction for /summarize
        if role == "assistant":
            self._last_interaction[user_id] = {
                "type": interaction_type,
                "content": content,
            }

    def get(self, user_id: int) -> list[dict]:
        """Get the conversation history for a user."""
        return list(self._store[user_id])

    def get_last_interaction(self, user_id: int) -> dict | None:
        """Get the last assistant response for a user."""
        return self._last_interaction.get(user_id)

    def clear(self, user_id: int) -> None:
        """Clear a user's history."""
        self._store.pop(user_id, None)
        self._last_interaction.pop(user_id, None)
