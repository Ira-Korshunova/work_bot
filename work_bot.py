#!/usr/bin/env python3
"""
🛠 Work Bot — Telegram бот для работы с документами, товарами и голосом
Кнопки: 📄 Документ, 📷 Товар, 🎤 Голос, ℹ️ Помощь

Интеграции:
  • Документы: Tesseract OCR (локально) — коносаменты, инвойсы, пакинг-листы
  • Товары: Qwen-VL через OpenRouter/DashScope
  • Голос: Yandex SpeechKit STT + TTS

Запуск:
    python3 work_bot.py

Требования:
    pip install python-telegram-bot python-dotenv requests Pillow pytesseract opencv-python-headless openai
    # macOS:
    brew install tesseract
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ── Загружаем .env ──
load_dotenv()

# ── Конфигурация ──
TELEGRAM_BOT_TOKEN = os.getenv("WORK_BOT_TOKEN", "").strip() or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
DASHSCOPE_BASE_URL = os.getenv("BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

# Пути к существующим агентам (задаются через переменные окружения)
DOC_AGENT_PATH = os.getenv("DOC_AGENT_PATH", "").strip()
VISION_AGENT_PATH = os.getenv("VISION_AGENT_PATH", "").strip()

if DOC_AGENT_PATH and os.path.isdir(DOC_AGENT_PATH):
    sys.path.insert(0, DOC_AGENT_PATH)
if VISION_AGENT_PATH and os.path.isdir(VISION_AGENT_PATH):
    sys.path.insert(0, VISION_AGENT_PATH)
# Свою директорию — в начало path, чтобы import rag_engine шёл из ДЗмод5_1
# (а не из copy_ДЗмод4_3_vision, где лежит устаревшая копия). Vision/Doc-агенты
# всё равно разрешатся через вставки выше.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Пути к бинарникам (Tesseract + Poppler) из переменных окружения ──
CONDA_BIN = os.getenv("CONDA_BIN", "").strip()
if CONDA_BIN and os.path.isdir(CONDA_BIN):
    os.environ["PATH"] = CONDA_BIN + os.pathsep + os.environ.get("PATH", "")
TESSDATA_PREFIX = os.getenv("TESSDATA_PREFIX", "").strip()
if TESSDATA_PREFIX and os.path.isdir(TESSDATA_PREFIX):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX

DOC_AGENT_AVAILABLE = False
VISION_AGENT_AVAILABLE = False
DocumentExtractor = None
VisionProductAgent = None

try:
    from document_extractor import DocumentExtractor
    import shutil
    if shutil.which("tesseract"):
        DOC_AGENT_AVAILABLE = True
    else:
        logging.warning("Tesseract не найден в PATH. Документы недоступны.")
except Exception as e:
    logging.warning(f"Document agent not available: {e}")

try:
    from agent_vision import VisionProductAgent
    VISION_AGENT_AVAILABLE = True
except Exception as e:
    logging.warning(f"Vision agent not available: {e}")

# ── RAG-движок (база знаний ВЭД) ──
RAG_AVAILABLE = False
rag_engine = None
try:
    import rag_engine
    RAG_AVAILABLE = rag_engine.RAG_AVAILABLE
except Exception as e:
    logging.warning(f"RAG engine not available: {e}")

# ── Логгер ──
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# httpx на INFO печатает URL запросов к Telegram API вида bot<ТОКЕН>/getMe —
# токен оседает в логе. Глушим до WARNING (только ошибки), без потери полезных логов бота.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── База данных ──
WORK_BOT_DB = os.path.join(os.path.dirname(__file__), "work_bot.db")


class WorkBotDB:
    """Логирование сессий и взаимодействий"""

    def __init__(self, db_path: str = WORK_BOT_DB):
        self.db_path = db_path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                type TEXT,
                input_summary TEXT,
                output_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def log_session(self, user_id: int, chat_id: int, mode: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (user_id, chat_id, mode) VALUES (?, ?, ?)",
            (user_id, chat_id, mode)
        )
        conn.commit()
        conn.close()

    def log_interaction(self, user_id: int, type_: str, input_summary: str, output_summary: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO interactions (user_id, type, input_summary, output_summary) VALUES (?, ?, ?, ?)",
            (user_id, type_, input_summary, output_summary)
        )
        conn.commit()
        conn.close()


# ── Менеджеры состояния и пользователей ──
from dialog_controller import SessionManager
from storage import UserDatabase

session_manager = SessionManager(session_timeout=3600)
user_db = UserDatabase(storage_path="./user_data.json")


# ── Query rewriting ──
def _rewrite_follow_up(q: str, history: list[dict]) -> str:
    """
    Раскрывает короткие уточняющие вопросы ('а сроки?', 'какие справки?')
    в самодостаточные запросы, используя предыдущие вопросы пользователя.
    Самодостаточные вопросы (например, '/top 3 что такое коносамент') не трогаем.
    """
    if not history:
        return q
    user_questions = [h["content"] for h in history[-6:] if h["role"] == "user"]
    if not user_questions:
        return q

    last_topic = user_questions[-1]
    q_lower = q.lower().strip()

    # команды /top — это всегда самостоятельный запрос, strip-аем префикс
    if q_lower.startswith("/top"):
        return q.strip()

    # явно самодостаточные вопросы
    if "что такое" in q_lower or "кто такой" in q_lower or "как определяется" in q_lower:
        return q.strip()

    # признаки уточняющего вопроса: короткий И содержит маркер
    follow_up_markers = (
        "а ", "а?", "какие", "какой", "чем", "где", "когда",
        "сколько", "почему", "кто", "сроки", "документы", "справки",
        "штрафы", "обязанности", "порядок", "размер", "ставка",
    )
    is_follow_up = (
        q_lower.startswith("а ")
        or (len(q.split()) <= 5 and any(marker in q_lower for marker in follow_up_markers))
    )
    if not is_follow_up:
        return q.strip()

    # простые эвристики раскрытия
    if q_lower.startswith("а "):
        return f"{last_topic.rstrip('?')} {q[2:].strip()}".capitalize()
    if any(w in q_lower for w in ("сроки", "срок", "дата", "когда")):
        return f"Какие сроки для {last_topic.rstrip('?').lower()}?"
    if any(w in q_lower for w in ("документы", "справки", "бумаги", "нужны")):
        return f"Какие документы нужны для {last_topic.rstrip('?').lower()}?"
    if any(w in q_lower for w in ("штрафы", "наказание", "ответственность")):
        return f"Какие штрафы/ответственность за {last_topic.rstrip('?').lower()}?"
    if any(w in q_lower for w in ("обязанности", "должен", "обязан")):
        return f"Какие обязанности у {last_topic.rstrip('?').lower()}?"

    # fallback: присоединяем к последней теме
    return f"{last_topic.rstrip('?')} {q.strip()}".capitalize()


# ── Клавиатура ──
def get_main_keyboard():
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📄 Документ"), KeyboardButton("📷 Товар")],
            [KeyboardButton("🎤 Голос"), KeyboardButton("ℹ️ Помощь")],
            [KeyboardButton("📚 База ВЭД")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


# ── Yandex SpeechKit ──
class YandexSpeechKit:
    """STT и TTS через Yandex SpeechKit"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def stt(self, audio_bytes: bytes, lang: str = "ru-RU") -> str:
        import requests

        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        params = {"topic": "general", "lang": lang}

        response = requests.post(url, headers=headers, params=params, data=audio_bytes, timeout=30)
        result = response.json()

        if "result" in result:
            return result["result"]
        elif "error_message" in result:
            raise RuntimeError(f"STT error: {result['error_message']}")
        else:
            raise RuntimeError(f"STT unknown response: {result}")

    def tts(self, text: str, voice: str = "filipp", lang: str = "ru-RU") -> bytes:
        import requests

        url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        data = {
            "text": text,
            "voice": voice,
            "lang": lang,
            "format": "oggopus",
            "sampleRateHertz": 48000,
        }

        response = requests.post(url, headers=headers, data=data, timeout=30)
        if response.status_code == 200:
            return response.content
        else:
            raise RuntimeError(f"TTS error {response.status_code}: {response.text}")


# ── Утилиты ──
def format_document_result(doc_type: str, data: dict) -> str:
    """Форматирование результата OCR в HTML"""
    type_display = {
        "bill_of_lading": "📄 Коносамент",
        "invoice": "🧾 Инвойс",
        "packing_list": "📦 Пакинг-лист",
        "unknown": "❓ Неизвестный тип",
    }.get(doc_type, doc_type)

    lines = [f"{type_display}", ""]

    key_names = {
        "bl_number": "Номер BL",
        "shipper_name": "Грузоотправитель",
        "shipper_address": "Адрес отправителя",
        "shipper_city_state_zip": "Город/шт/индекс отправителя",
        "consignee_name": "Грузополучатель",
        "consignee_address": "Адрес получателя",
        "consignee_city_state_zip": "Город/шт/индекс получателя",
        "notify_party": "Уведомлять",
        "port_of_loading": "Порт погрузки",
        "port_of_discharge": "Порт выгрузки",
        "vessel_name": "Судно",
        "voyage_number": "Рейс",
        "pro_number": "Pro номер",
        "container_numbers": "Контейнеры",
        "total_containers": "Всего контейнеров",
        "total_packages": "Всего мест",
        "weight": "Вес",
        "commodity": "Товар",
        "freight_charge_terms": "Условия фрахта",
        "date": "Дата",
        "pickup_date": "Дата забора",
        "special_instructions": "Спец. инструкции",
        "invoice_number": "Номер инвойса",
        "invoice_date": "Дата инвойса",
        "supplier": "Поставщик",
        "buyer": "Покупатель",
        "currency": "Валюта",
        "total_amount": "Сумма",
        "tax": "Налог",
        "payment_terms": "Условия оплаты",
        "packing_list_number": "Номер пакинг-листа",
        "shipper": "Отправитель",
        "consignee": "Получатель",
        "total_weight": "Общий вес",
        "total_volume": "Общий объем",
        "description": "Описание",
    }

    for key, label in key_names.items():
        val = data.get(key)
        if val is not None and val != "":
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            lines.append(f"<b>{label}:</b> {val}")

    if data.get("items"):
        lines.append("")
        lines.append(f"<b>Позиций:</b> {len(data['items'])}")

    text = "\n".join(lines)
    # Telegram limit ~4096
    if len(text) > 4000:
        text = text[:3900] + "\n\n..."
    return text


def format_product_result(data: dict) -> str:
    """Форматирование результата vision в HTML"""
    lines = ["📷 <b>Распознан товар</b>", ""]

    fields = [
        ("brand", "Бренд"),
        ("model_name", "Модель"),
        ("category", "Категория"),
        ("color", "Цвет"),
        ("material", "Материал"),
        ("dimensions", "Размеры"),
        ("weight", "Вес"),
        ("specs", "Характеристики"),
        ("description", "Описание"),
    ]

    for key, label in fields:
        val = data.get(key)
        if val:
            lines.append(f"<b>{label}:</b> {val}")

    price = data.get("price")
    currency = data.get("currency", "")
    if price:
        lines.insert(4, f"<b>Цена:</b> {price} {currency}")

    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:3900] + "\n\n..."
    return text


# ── Обработчики команд ──
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    rag_status = "🟢" if RAG_AVAILABLE else "🟡"
    welcome = (
        f"👋 Привет, {user.first_name}!\n\n"
        f"Я — рабочий бот. Выберите действие кнопкой ниже 👇\n\n"
        f"📄 Документ — OCR для коносаментов/инвойсов\n"
        f"📷 Товар — распознавание товара по фото\n"
        f"🎤 Голос — speech-to-text + text-to-speech\n"
        f"📚 База ВЭД — вопросы по нормативке ВЭД (RAG) {rag_status}"
    )
    await update.message.reply_text(welcome, reply_markup=get_main_keyboard())


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "🛠 <b>Рабочий бот — помощь</b>\n\n"
    text += "<b>📄 Документ</b> — отправьте фото или PDF документа (коносамент, инвойс, пакинг-лист). Бот распознает текст и извлечёт данные.\n\n"
    text += "<b>📷 Товар</b> — отправьте фото товара. Бот опишет бренд, модель, категорию, цену, характеристики.\n\n"
    text += "<b>🎤 Голос</b> — отправьте голосовое сообщение. Бот распознает речь в текст. Напишите текст — бот озвучит его.\n\n"
    text += "<b>📚 База ВЭД</b> — режим вопросов по нормативной базе ВЭД (валютный контроль, таможня, Инкотермс, документы сделки). Ответ строго по базе, со ссылкой на источник.\n\n"
    text += "<b>Команды:</b>\n"
    text += "  /start — перезапуск бота и показ клавиатуры\n"
    text += "  /help — эта справка\n"
    text += "  /ingest — индексация документов ВЭД (rag_data/docs)\n"
    text += "  /stats — статус базы ВЭД\n"
    text += "  /ask &lt;вопрос&gt; — разовый вопрос по базе ВЭД\n"
    text += "  /clear — очистить историю диалога\n"
    text += "  /top N — искать по N источникам (1–20)\n\n"
    text += "<b>Статус агентов:</b>\n"
    text += f"  {'✅' if DOC_AGENT_AVAILABLE else '⚠️'} Документы (OCR)\n"
    text += f"  {'✅' if VISION_AGENT_AVAILABLE else '⚠️'} Товары (Vision)\n"
    text += f"  {'✅' if YANDEX_API_KEY else '⚠️'} Голос (Yandex SpeechKit)\n"
    text += f"  {'✅' if RAG_AVAILABLE else '⚠️'} База ВЭД (RAG)\n\n"
    text += "<i>Если кнопки пропали — отправьте /start</i>"
    await update.message.reply_text(text, parse_mode="HTML", reply_markup=get_main_keyboard())


# ── Обработка кнопок и текста ──
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текст: кнопки ReplyKeyboard и обычный текст"""
    text = update.message.text
    user_id = update.effective_user.id

    # Кнопки главного меню
    session = session_manager.get_or_create_session(user_id)

    if text == "📄 Документ":
        session.state = "document"
        await update.message.reply_text(
            "📄 <b>Режим: Документ</b>\n\n"
            "Отправьте фото документа или PDF.\n"
            "Поддерживаются: коносамент, инвойс, пакинг-лист.",
            parse_mode="HTML"
        )
        return

    if text == "📷 Товар":
        session.state = "product"
        await update.message.reply_text(
            "📷 <b>Режим: Товар</b>\n\n"
            "Отправьте фото товара. Я опишу бренд, модель, категорию, цену и характеристики.",
            parse_mode="HTML"
        )
        return

    if text == "🎤 Голос":
        session.state = "voice"
        await update.message.reply_text(
            "🎤 <b>Режим: Голос</b>\n\n"
            "Отправьте голосовое сообщение — я распознаю текст.\n"
            "Напишите любой текст — я озвучу его голосом.",
            parse_mode="HTML"
        )
        return

    if text == "ℹ️ Помощь":
        await help_cmd(update, context)
        return

    if text == "📚 База ВЭД":
        session.state = "rag"
        if not RAG_AVAILABLE:
            await update.message.reply_text(
                "📚 <b>Режим: База ВЭД</b>\n\n"
                "⚠️ RAG-модуль недоступен. Проверьте API_KEY/BASE_URL в .env "
                "и что установлены faiss-cpu, openai, rank_bm25.",
                parse_mode="HTML")
            return
        await update.message.reply_text(
            "📚 <b>Режим: База ВЭД</b>\n\n"
            "Просто <b>напишите вопрос</b> — отвечу строго по базе со ссылкой на источник.\n\n"
            "Например: «Какие справки нужны для валютного контроля по контракту на 5 млн?»\n\n"
            "Кнопки ниже — управление базой. Любая другая кнопка клавиатуры — выход из режима.",
            parse_mode="HTML",
            reply_markup=rag_inline_keyboard())
        return

    # Режим RAG: свободный текст — вопрос по базе ВЭД
    if session.state == "rag":
        await handle_rag_qa(update, context)
        return

    # Если режим голоса и пришёл текст — озвучиваем
    if session.state == "voice":
        await handle_tts(update, context)
        return

    # По умолчанию
    await update.message.reply_text(
        "Выберите действие кнопкой ниже 👇",
        reply_markup=get_main_keyboard()
    )


# ── Обработка фото ──
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    session = session_manager.get_or_create_session(user_id)

    if session.state == "document":
        await handle_document_photo(update, context)
    elif session.state == "product":
        await handle_product_photo(update, context)
    else:
        # Если режим не выбран — предлагаем выбрать
        await update.message.reply_text(
            "📷 Фото получено. Что это?\n\n"
            "Выберите режим в меню ниже 👇",
            reply_markup=get_main_keyboard()
        )
        # Запомним, что фото ждёт выбора режима
        session.set_metadata("pending_photo", update.message.photo[-1].file_id)
        session.state = "pending_photo"


async def handle_document_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    await update.message.reply_text("🔍 Распознаю документ...")

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        extractor = DocumentExtractor()
        doc_type, data = await asyncio.to_thread(extractor.extract_from_image, tmp_path)

        # Проверка: это действительно документ?
        raw_text = data.get("raw_text", "")
        text_lower = raw_text.lower()

        # Ключевые слова документов
        doc_keywords = [
            "invoice", "bill of lading", "b/l", "packing list", "consignee",
            "shipper", "port of loading", "port of discharge", "vessel", "voyage",
            "container", "bl number", "pro number", "freight", "commodity",
            "usd", "eur", "руб", "total amount", "grand total", "tax", "vat",
            "payment terms", "bank", "iban", "swift", "amount due"
        ]
        # Ключевые слова товаров (если найдены — точно не документ)
        product_keywords = [
            "микроволнов", "холодильник", "телевизор", "наушник", "кроссов",
            "пылесос", "фотоаппарат", "миксер", "блендер", "стиральн", "посудомоечн",
            "плита", "духовк", "гриль", "утюг", "паров", "обогреватель", "вентилятор",
            "ноутбук", "телефон", "планшет", "часы", "пылесос", "робот",
            "артикул", "код товара", "в корзину", "отзыв", "рейтинг", "звезд",
            "gorenje", "samsung", "lg", "bosch", "philips", "xioami", "brand"
        ]

        has_doc_kw = any(kw in text_lower for kw in doc_keywords)
        has_product_kw = any(kw in text_lower for kw in product_keywords)

        # Точно документ, если тип определился
        is_likely_document = doc_type != "unknown"
        # Или много текста + есть документные ключевые слова (скрин товара даст много текста, но без doc_keywords)
        is_likely_document = is_likely_document or (len(raw_text) > 100 and has_doc_kw)

        # Если явно товар — это не документ
        if not is_likely_document and has_product_kw:
            await update.message.reply_text(
                "🤔 Это не похоже на документ (коносамент, инвойс или пакинг-лист).\n\n"
                "Похоже, здесь изображён товар!\n\n"
                "📷 Нажмите <b>Товар</b> — я распознаю бренд, модель и цену.",
                parse_mode="HTML",
                reply_markup=get_main_keyboard(),
            )
            # Сбрасываем режим, чтобы можно было нажать Товар
            session.set_metadata("pending_photo", photo.file_id)
            session.state = "pending_photo"
            return

        if not is_likely_document:
            await update.message.reply_text(
                "🤔 Это не похоже на документ (коносамент, инвойс или пакинг-лист).\n\n"
                "Возможно, это фото товара? Нажмите 📷 <b>Товар</b>, чтобы распознать бренд, модель и цену.\n\n"
                "Или отправьте другой документ.",
                parse_mode="HTML",
                reply_markup=get_main_keyboard(),
            )
            session.set_metadata("pending_photo", photo.file_id)
            session.state = "pending_photo"
            return

        text = format_document_result(doc_type, data)
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=get_main_keyboard())

        # Дружелюбное приглашение продолжить
        await update.message.reply_text(
            "✨ Документ обработан! Что дальше? Выберите действие в меню ниже 👇",
            reply_markup=get_main_keyboard(),
        )

        db = WorkBotDB()
        db.log_interaction(user_id, "document", f"photo:{photo.file_id}", doc_type)

    except Exception as e:
        logger.error(f"Document OCR error: {e}", exc_info=True)
        await update.message.reply_text(
            "🤔 Не удалось распознать документ.\n\n"
            "Возможно, это фото товара? Нажмите 📷 Товар, чтобы распознать бренд, модель и цену.",
            reply_markup=get_main_keyboard(),
        )
        # Сбрасываем режим
        session = session_manager.get_or_create_session(user_id)
        session.set_metadata("pending_photo", photo.file_id)
        session.state = "pending_photo"

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def handle_product_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if not VISION_AGENT_AVAILABLE or not VisionProductAgent:
        await update.message.reply_text(
            "❌ Агент товаров недоступен. Проверьте папку ~/Desktop/Домашка/ДЗмод4_3_vision"
        )
        return

    if not OPENROUTER_API_KEY:
        await update.message.reply_text("❌ Нет API_KEY для Vision API. Добавьте в .env")
        return

    await update.message.reply_text("🔍 Анализирую товар...")

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        # Используем DashScope International для vision
        agent = VisionProductAgent(
            api_key=OPENROUTER_API_KEY,
            model="qwen3-vl-flash",
        )
        # Переопределяем base_url на dashscope-intl
        agent.client.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        data = await asyncio.to_thread(agent.analyze_product, tmp_path)

        if "error" in data and not data.get("brand"):
            await update.message.reply_text(f"❌ Ошибка анализа: {data['error']}")
            return

        # Сохраняем в оригинальную БД продуктов
        try:
            await asyncio.to_thread(agent.save_product, data)
        except Exception as e:
            logger.warning(f"Save product failed: {e}")

        text = format_product_result(data)
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=get_main_keyboard())

        # Дружелюбное приглашение продолжить
        await update.message.reply_text(
            "✨ Готово! Что дальше? Выберите действие в меню ниже 👇",
            reply_markup=get_main_keyboard(),
        )

        db = WorkBotDB()
        db.log_interaction(user_id, "product", f"photo:{photo.file_id}", data.get("description", "")[:100])

    except Exception as e:
        logger.error(f"Vision error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка анализа: {e}")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Обработка документов (PDF) ──
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка PDF и других файлов-документов"""
    user_id = update.effective_user.id
    document = update.message.document

    if not document:
        return

    # Проверяем расширение
    filename = document.file_name or ""
    ext = Path(filename).suffix.lower()

    if ext not in {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        await update.message.reply_text(
            "❌ Формат не поддерживается. Отправьте PDF или изображение.",
            reply_markup=get_main_keyboard(),
        )
        return

    # Мягкий fallback если OCR недоступен
    if not DOC_AGENT_AVAILABLE or not DocumentExtractor:
        await update.message.reply_text(
            "🤔 Распознавание документов сейчас недоступно.\n\n"
            "Возможно, это фото товара или документ с товаром?\n"
            "Попробуйте отправить это же изображение через 📷 <b>Товар</b> — "
            "бот опишет содержимое и извлечёт данные.",
            parse_mode="HTML",
            reply_markup=get_main_keyboard(),
        )
        await update.message.reply_text(
            "✨ Что дальше? Выберите действие в меню ниже 👇",
            reply_markup=get_main_keyboard(),
        )
        return

    await update.message.reply_text("🔍 Распознаю документ...")

    file = await context.bot.get_file(document.file_id)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        extractor = DocumentExtractor()
        doc_type, data = await asyncio.to_thread(extractor.extract_from_image, tmp_path)

        # Проверка: это действительно документ?
        raw_text = data.get("raw_text", "")
        text_lower = raw_text.lower()

        doc_keywords = [
            "invoice", "bill of lading", "b/l", "packing list", "consignee",
            "shipper", "port of loading", "port of discharge", "vessel", "voyage",
            "container", "bl number", "pro number", "freight", "commodity",
            "usd", "eur", "руб", "total amount", "grand total", "tax", "vat",
            "payment terms", "bank", "iban", "swift", "amount due"
        ]
        product_keywords = [
            "микроволнов", "холодильник", "телевизор", "наушник", "кроссов",
            "пылесос", "фотоаппарат", "миксер", "блендер", "стиральн", "посудомоечн",
            "плита", "духовк", "гриль", "утюг", "паров", "обогреватель", "вентилятор",
            "ноутбук", "телефон", "планшет", "часы", "пылесос", "робот",
            "артикул", "код товара", "в корзину", "отзыв", "рейтинг", "звезд",
            "gorenje", "samsung", "lg", "bosch", "philips", "xioami", "brand"
        ]

        has_doc_kw = any(kw in text_lower for kw in doc_keywords)
        has_product_kw = any(kw in text_lower for kw in product_keywords)
        is_likely_document = (doc_type != "unknown") or (len(raw_text) > 100 and has_doc_kw)

        if not is_likely_document and has_product_kw:
            await update.message.reply_text(
                "🤔 Это не похоже на документ (коносамент, инвойс или пакинг-лист).\n\n"
                "Похоже, здесь изображён товар!\n\n"
                "📷 Нажмите <b>Товар</b> — я распознаю бренд, модель и цену.",
                parse_mode="HTML",
                reply_markup=get_main_keyboard(),
            )
            session.set_metadata("pending_photo", document.file_id)
            session.state = "pending_photo"
            return

        if not is_likely_document:
            await update.message.reply_text(
                "🤔 Это не похоже на документ (коносамент, инвойс или пакинг-лист).\n\n"
                "Возможно, это фото товара? Нажмите 📷 <b>Товар</b>, чтобы распознать бренд, модель и цену.\n\n"
                "Или отправьте другой документ.",
                parse_mode="HTML",
                reply_markup=get_main_keyboard(),
            )
            # Сбрасываем режим, чтобы не застрять
            session.set_metadata("pending_photo", document.file_id)
            session.state = "pending_photo"
            return

        text = format_document_result(doc_type, data)
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=get_main_keyboard())

        # Дружелюбное приглашение продолжить
        await update.message.reply_text(
            "✨ Документ обработан! Что дальше? Выберите действие в меню ниже 👇",
            reply_markup=get_main_keyboard(),
        )

        db = WorkBotDB()
        db.log_interaction(user_id, "document", f"doc:{filename}", doc_type)

    except Exception as e:
        logger.error(f"Document error: {e}", exc_info=True)
        await update.message.reply_text(
            "🤔 Не удалось распознать документ.\n\n"
            "Возможно, это фото товара? Нажмите 📷 Товар, чтобы распознать бренд, модель и цену.",
            reply_markup=get_main_keyboard(),
        )
        # Сбрасываем режим
        session = session_manager.get_or_create_session(user_id)
        session.set_metadata("pending_photo", document.file_id)
        session.state = "pending_photo"

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Голос ──
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    logger.info(f"🎤 VOICE received from user {user_id}, voice_id={update.message.voice.file_id if update.message.voice else 'none'}")

    if not YANDEX_API_KEY:
        await update.message.reply_text(
            "❌ Голосовые функции недоступны.\n"
            "Добавьте YANDEX_API_KEY в .env файл.\n"
            "Ключ можно получить в Yandex Cloud: https://cloud.yandex.ru/"
        )
        return

    await update.message.reply_text("🎤 Распознаю речь...")

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        speech = YandexSpeechKit(YANDEX_API_KEY)
        text = await asyncio.to_thread(speech.stt, audio_bytes)

        await update.message.reply_text(
            f"📝 <b>Распознано:</b>\n<i>{text}</i>\n\n"
            f"Напишите текст — я озвучу его.",
            parse_mode="HTML",
            reply_markup=get_main_keyboard(),
        )

        session = session_manager.get_session(user_id) or session_manager.get_or_create_session(user_id)
        session.state = "voice"
        session.set_metadata("last_text", text)

        db = WorkBotDB()
        db.log_interaction(user_id, "voice_stt", f"voice:{voice.file_id}", text[:200])

    except Exception as e:
        logger.error(f"STT error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка распознавания: {e}")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


async def handle_tts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Озвучка текста (только в режиме голоса или по команде)"""
    user_id = update.effective_user.id
    text = update.message.text

    if not YANDEX_API_KEY:
        await update.message.reply_text("❌ YANDEX_API_KEY не настроен.")
        return

    if not text.strip():
        return

    await update.message.reply_text("🔊 Синтезирую речь...")

    try:
        speech = YandexSpeechKit(YANDEX_API_KEY)
        audio = await asyncio.to_thread(speech.tts, text[:1000])  # ограничение по длине

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as voice_file:
            await update.message.reply_voice(voice=voice_file)

        os.unlink(tmp_path)

        # Дружелюбное приглашение продолжить
        await update.message.reply_text(
            "✨ Готово! Что дальше? Выберите действие в меню ниже 👇",
            reply_markup=get_main_keyboard(),
        )

        db = WorkBotDB()
        db.log_interaction(user_id, "voice_tts", text[:100], "voice_sent")

    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка синтеза: {e}")


# ── RAG (база знаний ВЭД) — хендлеры ──
def rag_inline_keyboard() -> InlineKeyboardMarkup:
    """Inline-кнопки управления базой ВЭД (показываются в режиме База ВЭД)."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Индексировать базу", callback_data="rag_ingest"),
         InlineKeyboardButton("📊 Статус базы", callback_data="rag_stats")],
        [InlineKeyboardButton("❌ Выйти из режима", callback_data="rag_exit")],
    ])


async def rag_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка inline-кнопок режима База ВЭД."""
    q = update.callback_query
    await q.answer()
    data = q.data
    user_id = q.from_user.id

    if data == "rag_exit":
        session_manager.delete_session(user_id)
        try:
            await q.edit_message_text(
                "Вышли из режима «База ВЭД».\nВыберите действие на основной клавиатуре.")
        except Exception:
            await q.message.reply_text("Вышли из режима «База ВЭД».")
        return

    if not RAG_AVAILABLE:
        await q.edit_message_text("❌ RAG-модуль недоступен. Проверьте API_KEY/BASE_URL и библиотеки.")
        return

    if data == "rag_ingest":
        await q.edit_message_text("⏳ Индексирую базу ВЭД…\nЭто может занять до минуты.")
        try:
            result = await asyncio.to_thread(rag_engine.ingest, True)
        except Exception as e:
            logger.error(f"RAG ingest error: {e}", exc_info=True)
            await q.edit_message_text(f"❌ Ошибка индексации: {e}")
            return
        files = "\n".join(f"  • {f}" for f in result.get("files", [])) or "  (нет файлов)"
        text = (
            "📚 <b>Индексация завершена</b>\n\n"
            f"🆕 Добавлено: {result['added']}\n"
            f"♻️ Обновлено: {result['updated']}\n"
            f"🗑 Удалено: {result['removed']}\n"
            f"🧩 Всего чанков: {result['total_chunks']}\n\n"
            f"<b>Файлы:</b>\n{files}"
        )
        errs = result.get("errors") or []
        if errs:
            text += "\n\n<b>Ошибки:</b>\n" + "\n".join(f"  • {e}" for e in errs)
        try:
            await q.edit_message_text(text, parse_mode="HTML", reply_markup=rag_inline_keyboard())
        except Exception:
            await q.message.reply_text(text, parse_mode="HTML")
        WorkBotDB().log_interaction(user_id, "rag_ingest", "", str(result['total_chunks']))
        return

    if data == "rag_stats":
        stats = await asyncio.to_thread(rag_engine.get_stats)
        if stats.get("empty"):
            await q.edit_message_text(
                "📚 <b>База ВЭД пуста.</b>\n\n"
                f"Положите документы (.txt/.md/.pdf) в:\n<code>{stats['docs_dir']}</code>\n"
                "и нажмите «🔄 Индексировать базу».",
                parse_mode="HTML", reply_markup=rag_inline_keyboard())
            return
        files = "\n".join(f"  • {f['name']} — {f['chunks']} чанков" for f in stats.get("files", [])) or "  (нет)"
        text = (
            "📊 <b>Статус базы ВЭД</b>\n\n"
            f"🧩 Чанков: {stats['collection_count']}\n"
            f"🧠 Эмбеддинги: {stats['models']['embed']}\n"
            f"🧠 Чат: {stats['models']['chat']}\n"
            f"🔍 Реранкер: {stats.get('rerank_on')}\n"
            f"🔎 BM25: {stats.get('bm25')}\n\n"
            f"<b>Файлы:</b>\n{files}"
        )
        try:
            await q.edit_message_text(text, parse_mode="HTML", reply_markup=rag_inline_keyboard())
        except Exception:
            await q.message.reply_text(text, parse_mode="HTML")
        return


async def _send_long(msg, update, text, parse_mode="HTML"):
    """Отправка длинного HTML-текста с разбивкой по лимиту Telegram (~4096)."""
    LIMIT = 4000
    chunks = [text[i:i + LIMIT] for i in range(0, len(text), LIMIT)] or [text]
    try:
        await msg.edit_text(chunks[0], parse_mode=parse_mode)
    except Exception:
        await update.message.reply_text(chunks[0], parse_mode=parse_mode)
    for c in chunks[1:]:
        await update.message.reply_text(c, parse_mode=parse_mode)



async def rag_ingest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not RAG_AVAILABLE:
        await update.message.reply_text(
            "❌ RAG-модуль недоступен. Проверьте API_KEY/BASE_URL и библиотеки (faiss, openai, rank_bm25).")
        return
    msg = await update.message.reply_text("⏳ <b>Индексирую базу ВЭД…</b>\nЭто может занять до минуты.", parse_mode="HTML")
    try:
        result = await asyncio.to_thread(rag_engine.ingest, True)
    except Exception as e:
        logger.error(f"RAG ingest error: {e}", exc_info=True)
        await msg.edit_text(f"❌ Ошибка индексации: {e}")
        return
    files = "\n".join(f"  • {f}" for f in result.get("files", [])) or "  (нет файлов)"
    text = (
        "📚 <b>Индексация базы ВЭД завершена</b>\n\n"
        f"🆕 Добавлено файлов: {result['added']}\n"
        f"♻️ Обновлено: {result['updated']}\n"
        f"🗑 Удалено: {result['removed']}\n"
        f"🧩 Всего чанков: {result['total_chunks']}\n\n"
        f"<b>Файлы:</b>\n{files}"
    )
    errs = result.get("errors") or []
    if errs:
        text += "\n\n<b>Ошибки:</b>\n" + "\n".join(f"  • {e}" for e in errs)
    await _send_long(msg, update, text)
    WorkBotDB().log_interaction(user_id, "rag_ingest", "", str(result['total_chunks']))


async def rag_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not RAG_AVAILABLE:
        await update.message.reply_text("❌ RAG-модуль недоступен.")
        return
    stats = await asyncio.to_thread(rag_engine.get_stats)
    if stats.get("empty"):
        await update.message.reply_text(
            "📚 <b>База ВЭД пуста.</b>\n\n"
            f"Положите документы (.txt/.md/.pdf) в:\n<code>{stats['docs_dir']}</code>\nи выполните /ingest",
            parse_mode="HTML")
        return
    files = "\n".join(f"  • {f['name']} — {f['chunks']} чанков" for f in stats.get("files", [])) or "  (нет)"
    text = (
        "📊 <b>Статус базы ВЭД</b>\n\n"
        f"🧩 Чанков в базе: {stats['collection_count']}\n"
        f"🧠 Эмбеддинги: {stats['models']['embed']}\n"
        f"🧠 Чат: {stats['models']['chat']}\n"
        f"🔍 Реранкер: {stats.get('rerank_on')}\n"
        f"🔎 BM25: {stats.get('bm25')}\n\n"
        f"<b>Файлы:</b>\n{files}"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def handle_rag_qa(update: Update, context: ContextTypes.DEFAULT_TYPE, question: str = None) -> None:
    user_id = update.effective_user.id
    if not RAG_AVAILABLE:
        await update.message.reply_text(
            "❌ RAG-модуль недоступен. Проверьте API_KEY/BASE_URL и библиотеки (faiss, openai, rank_bm25).")
        return
    q = question if question is not None else update.message.text
    if not q or not q.strip():
        await update.message.reply_text("Задайте вопрос по ВЭД.")
        return

    # Загружаем/создаём сессию пользователя
    session = session_manager.get_or_create_session(user_id)
    session.state = "rag"
    history = session.get_conversation_history(max_messages=10)

    # Сохраняем факт о пользователе, если он назвал имя
    q_lower = q.strip().lower()
    if "меня зовут" in q_lower:
        # Простая эвристика: "Меня зовут Ирина"
        parts = q.strip().split()
        for i, word in enumerate(parts):
            if word.lower() in ("зовут", "зовут.") and i + 1 < len(parts):
                name = parts[i + 1].strip(",.!?")
                if name:
                    user_db.create_or_update_user(user_id, name=name)
                    user_db.set_fact(user_id, "name", name)
                break

    # Долгосрочная память: извлекаем известные факты о пользователе
    user_facts = user_db.get_facts(user_id) or {}
    user_profile = ""
    if user_facts:
        profile_lines = [f"- {k}: {v}" for k, v in user_facts.items()]
        user_profile = "\n".join(profile_lines)

    msg = await update.message.reply_text("⏳ Ищу в базе ВЭД…")
    logger.info(f"handle_rag_qa: user={user_id}, top_k={session.top_k}, q={q.strip()[:80]}")

    # top_k: персональный или по умолчанию
    top_k = session.top_k

    # Переписываем короткий follow-up в самодостаточный вопрос на основе истории.
    rewritten_q = _rewrite_follow_up(q.strip(), history)

    # Формируем запрос для retrieval: только текущий (возможно, раскрытый) вопрос.
    # История диалога передаётся в rag_engine.query отдельно, чтобы не путать
    # поисковый движок предыдущими темами, а LLM видел контекст диалога.
    expanded_q = rewritten_q
    if user_profile:
        expanded_q = (
            "Известные факты о пользователе:\n"
            + "\n".join(f"- {line}" for line in user_profile.splitlines())
            + f"\n\nТекущий вопрос: {rewritten_q}"
        )

    logger.info(f"handle_rag_qa: запуск rag_engine.query с k={top_k}, q={expanded_q[:80]}")
    try:
        # Передаём expanded_q для retrieval и history для LLM-контекста.
        result = await asyncio.to_thread(rag_engine.query, expanded_q, k=top_k, history=history)
        logger.info(f"handle_rag_qa: rag_engine.query завершён, answer_len={len(result.get('answer', ''))}")
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        await msg.edit_text(f"⏱️/❌ Ошибка: {e}")
        return
    answer = result.get("answer", "") or ""
    sources = result.get("sources", []) or []
    method = result.get("method", "")
    method_tag = {"rrf": "BM25+dense+RRF", "rerank": "RRF+rerank", "empty": "пусто",
                  "error": "ошибка", "none": "—"}.get(method, method)

    # Если бот честно не нашёл ответ — не показываем слабые источники
    not_found = result.get("not_found") or result.get("empty") or "не нашёл" in answer.lower()
    src_block = ""
    if sources and not not_found:
        src_lines = []
        for i, s in enumerate(sources, 1):
            h = s.get("heading")
            snippet = s.get("snippet", "")
            label = f"{s['source']}"
            if h:
                label += f" — {h}"
            elif snippet:
                label += f" — {snippet[:70]}..."
            src_lines.append(f"  {i}. {label}")
        src_block = "\n\n📚 <b>Источники:</b>\n" + "\n".join(src_lines)

    footer = f"\n\n<i>Метод: {method_tag} • Источников в контексте: {len(sources)} • Не является юридической консультацией.</i>"
    text = f"📚 <b>Ответ по базе ВЭД</b>\n\n{answer}{src_block}{footer}"
    await _send_long(msg, update, text)

    # Сохраняем вопрос и ответ в историю
    session.add_message("user", q.strip())
    session.add_message("assistant", answer)

    WorkBotDB().log_interaction(user_id, "rag_query", q[:200], answer[:300])


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Очистка истории диалога."""
    user_id = update.effective_user.id
    session = session_manager.get_or_create_session(user_id)
    session.clear_conversation_history()
    session.top_k = 5
    await update.message.reply_text(
        "✅ История диалога очищена!\n\n"
        "Я больше не помню предыдущие сообщения в этом чате.",
        reply_markup=get_main_keyboard()
    )


async def top_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Установка персонального top_k и сразу ответ на вопрос (если указан)."""
    user_id = update.effective_user.id
    args = context.args
    logger.info(f"/top вызван пользователем {user_id}, args={args}")

    if not args or not args[0].isdigit():
        await update.message.reply_text(
            "ℹ️ Использование: /top N [вопрос]\n"
            "Например: <code>/top 3 Что такое коносамент?</code>\n"
            "Или просто: <code>/top 3</code>, а затем задайте вопрос.\n"
            "Диапазон: 1–20.",
            parse_mode="HTML",
            reply_markup=get_main_keyboard())
        return

    top_k = int(args[0])
    if top_k < 1 or top_k > 20:
        await update.message.reply_text("❌ Укажите число от 1 до 20.")
        return

    session = session_manager.get_or_create_session(user_id)
    session.top_k = top_k
    session.state = "rag"

    # Если после /top N есть вопрос — сразу на него отвечаем
    if len(args) >= 2:
        question = " ".join(args[1:])
        logger.info(f"/top {top_k} с вопросом: {question}")
        await handle_rag_qa(update, context, question=question)
        return

    # Иначе просто меняем настройку
    await update.message.reply_text(
        f"🔍 В режиме «База ВЭД» буду искать по <b>{top_k}</b> источникам.\n\n"
        f"Задайте вопрос по ВЭД.",
        parse_mode="HTML",
        reply_markup=rag_inline_keyboard()
    )
    logger.info(f"/top ответ отправлен {user_id}")


async def rag_ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = " ".join(context.args) if context.args else ""
    if not q.strip():
        await update.message.reply_text(
            "Использование:\n<code>/ask &lt;вопрос по ВЭД&gt;</code>\n\n"
            "Например: <code>/ask Какие справки нужны для валютного контроля?</code>",
            parse_mode="HTML")
        return
    await handle_rag_qa(update, context, question=q)


# ── Main ──
def main():
    if not TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN (или WORK_BOT_TOKEN) не найден в .env")
        sys.exit(1)

    # Инициализируем БД
    WorkBotDB()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Команды
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("ingest", rag_ingest_cmd))
    application.add_handler(CommandHandler("stats", rag_stats_cmd))
    application.add_handler(CommandHandler("ask", rag_ask_cmd))
    application.add_handler(CommandHandler("clear", clear_cmd))
    application.add_handler(CommandHandler("top", top_cmd))

    # Текст (кнопки + обычный текст)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Inline-кнопки режима База ВЭД
    application.add_handler(CallbackQueryHandler(rag_callback, pattern="^rag_"))

    # Фото
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    # Документы (PDF)
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    # Голос
    application.add_handler(MessageHandler(filters.VOICE, voice_handler))

    logger.info("🚀 Work Bot запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
