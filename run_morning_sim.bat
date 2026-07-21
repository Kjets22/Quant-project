@echo off
rem run_morning_sim.bat — daily QQQ morning-strategy simulation + improvement search
rem (scheduled 16:35 Mon-Fri; morning_daily.py does its own git commit/push).
cd /d C:\Users\kjets\capture_trader
python morning_daily.py >> runs\morning_task.log 2>&1
