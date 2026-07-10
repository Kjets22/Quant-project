@echo off
rem run_daily_report.bat — end-of-day report + push to the repo (scheduled 16:15 Mon-Fri).
cd /d C:\Users\kjets\capture_trader
python daily_report.py >> runs\alpaca_task.log 2>&1
git add runs/daily_reports runs/alpaca2_ledger.json runs/alpaca_log.txt >nul 2>&1
git commit -m "Daily A/B report" >nul 2>&1
git push origin main >nul 2>&1
