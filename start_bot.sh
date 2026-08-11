#!/bin/zsh
# Автозапуск DiV_executive бота с ожиданием интернета.
# Локальные пути и ключи задаются в .env (см. README, файл .env в .gitignore).

set -a
[ -f .env ] && source .env
set +a

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BOT_DIR"

# ── Пути к бинарникам (Tesseract + Poppler) из переменных окружения ──
if [ -n "$CONDA_BIN" ]; then
    export PATH="$CONDA_BIN:$PATH"
fi
if [ -n "$TESSDATA_PREFIX" ]; then
    export TESSDATA_PREFIX
fi

# Пути к агентам (папки с document_extractor / vision-агентом)
PYTHONPATH="$BOT_DIR"
[ -n "$DOC_AGENT_PATH" ] && PYTHONPATH="$PYTHONPATH:$DOC_AGENT_PATH"
[ -n "$VISION_AGENT_PATH" ] && PYTHONPATH="$PYTHONPATH:$VISION_AGENT_PATH"
export PYTHONPATH

# Ждём интернет (Telegram API)
for i in {1..30}; do
    if curl -s --max-time 3 "https://api.telegram.org" > /dev/null; then
        break
    fi
    echo "$(date): жду интернет..." >> bot_launchd.out
    sleep 2
done

echo "$(date): запускаю work_bot.py" >> bot_launchd.out
exec python3 work_bot.py >> bot_launchd.out 2>> bot_launchd.err
