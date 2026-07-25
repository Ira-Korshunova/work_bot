"""
Контекст пользователя.
Хранит историю диалога, метаданные и состояние сессии.
Адаптирован под work_bot.py.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class UserContext:
    """Контекст диалога одного пользователя."""

    def __init__(self, user_id: str | int):
        self.user_id = str(user_id)
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.conversation_history: List[Dict[str, str]] = []
        self.metadata: Dict[str, Any] = {}
        self.state: str = "active"
        self.message_count: int = 0
        self.top_k: int = 5  # персональный top_k, можно менять через /top

    def add_message(self, role: str, content: str):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.message_count += 1
        self.update_last_activity()

    def get_conversation_history(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        history = self.conversation_history
        if max_messages:
            history = history[-max_messages:]
        return [{"role": msg["role"], "content": msg["content"]} for msg in history]

    def clear_conversation_history(self):
        self.conversation_history = []
        logger.info(f"История диалога очищена для {self.user_id}")

    def update_last_activity(self):
        self.last_activity = datetime.now()

    def is_expired(self, timeout_seconds: int) -> bool:
        return datetime.now() - self.last_activity > timedelta(seconds=timeout_seconds)

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "conversation_length": len(self.conversation_history),
            "state": self.state,
            "metadata": self.metadata,
            "top_k": self.top_k,
        }
