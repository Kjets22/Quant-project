@echo off
rem run_alpaca_bot.bat — one bot cycle. Scheduled via Task Scheduler (see ALPACA_BOT_GUIDE.md).
cd /d C:\Users\kjets\capture_trader
python alpaca_bot2.py --once >> runs\alpaca_task.log 2>&1
