"""
База данных пользователей.
Сохраняет персональные факты для долгосрочной памяти.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class UserDatabase:
    """Простая файловая БД пользователей (JSON)."""

    def __init__(self, storage_path: str = "./user_data.json"):
        self.storage_path = storage_path
        self.users: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки user_db: {e}")
                return {}
        return {}

    def _save(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения user_db: {e}")

    def create_or_update_user(
        self,
        user_id: str | int,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        user_id = str(user_id)
        now = datetime.now().isoformat()
        if user_id not in self.users:
            self.users[user_id] = {
                "id": user_id,
                "name": name,
                "created_at": now,
                "last_active": now,
                "message_count": 0,
                "preferences": {},
                "facts": {},
                "metadata": metadata or {},
            }
            logger.info(f"Создан пользователь {user_id}")
        else:
            if name:
                self.users[user_id]["name"] = name
            self.users[user_id]["last_active"] = now
            if metadata:
                self.users[user_id]["metadata"].update(metadata)
        self._save()

    def increment_message_count(self, user_id: str | int):
        user_id = str(user_id)
        if user_id in self.users:
            self.users[user_id]["message_count"] += 1
            self.users[user_id]["last_active"] = datetime.now().isoformat()
            self._save()

    def set_fact(self, user_id: str | int, key: str, value: Any):
        """Сохраняет факт о пользователе для долгосрочной памяти."""
        user_id = str(user_id)
        if user_id not in self.users:
            self.create_or_update_user(user_id)
        self.users[user_id].setdefault("facts", {})[key] = value
        self._save()

    def get_facts(self, user_id: str | int) -> Dict[str, Any]:
        user_id = str(user_id)
        if user_id not in self.users:
            return {}
        return self.users[user_id].get("facts", {})

    def get_name(self, user_id: str | int) -> Optional[str]:
        user_id = str(user_id)
        return self.users.get(user_id, {}).get("name")

    def get_user_count(self) -> int:
        return len(self.users)

    def get_all_users(self) -> Dict[str, Dict[str, Any]]:
        return self.users
