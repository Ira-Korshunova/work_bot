"""
Менеджер сессий пользователей.
Управляет UserContext: создание, получение, очистка, таймаут.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

from .user_context import UserContext

logger = logging.getLogger(__name__)


class SessionManager:
    """Менеджер сессий пользователей."""

    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, UserContext] = {}
        self.session_timeout = session_timeout

    def get_or_create_session(self, user_id: str | int) -> UserContext:
        user_id = str(user_id)
        session = self.sessions.get(user_id)
        if session and not session.is_expired(self.session_timeout):
            session.update_last_activity()
            return session
        if session:
            logger.info(f"Сессия {user_id} истекла, создаём новую")
        session = UserContext(user_id)
        self.sessions[user_id] = session
        return session

    def get_session(self, user_id: str | int) -> Optional[UserContext]:
        user_id = str(user_id)
        return self.sessions.get(user_id)

    def delete_session(self, user_id: str | int):
        user_id = str(user_id)
        if user_id in self.sessions:
            del self.sessions[user_id]
            logger.info(f"Сессия {user_id} удалена")

    def cleanup_expired_sessions(self):
        expired = [uid for uid, s in self.sessions.items() if s.is_expired(self.session_timeout)]
        for uid in expired:
            del self.sessions[uid]
            logger.info(f"Удалена истекшая сессия: {uid}")

    def get_active_session_count(self) -> int:
        self.cleanup_expired_sessions()
        return len(self.sessions)
