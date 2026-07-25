#!/bin/zsh
# Автозапуск DiV_executive бота с ожиданием интернета

cd "/Users/irina/Documents/Домашка/ДЗмод5_1"
export PATH="/Users/irina/miniconda3/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/usr/local/bin:/usr/bin:/bin"
export TESSDATA_PREFIX="/Users/irina/miniconda3/share/tessdata"
export PYTHONPATH="/Users/irina/Documents/Домашка/ДЗмод5_1:/Users/irina/Desktop/Домашка/ДЗмод4_3"

# Ждём интернет (Telegram API)
for i in {1..30}; do
    if curl -s --max-time 3 "https://api.telegram.org" > /dev/null; then
        break
    fi
    echo "$(date): жду интернет..." >> bot_launchd.out
    sleep 2
done

echo "$(date): запускаю work_bot.py" >> bot_launchd.out
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -c "
import sys
sys.path.insert(0, '/Users/irina/Desktop/Домашка/ДЗмод4_3')
import work_bot
work_bot.main()
" >> bot_launchd.out 2>> bot_launchd.err
