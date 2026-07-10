' run_alpaca_hidden.vbs — runs one bot cycle with NO console window.
' A visible window can be closed mid-cycle (which aborts Python/MKL with forrtl 200);
' running hidden removes that failure mode entirely. Output still goes to
' runs\alpaca_task.log via the .bat redirection.
CreateObject("Wscript.Shell").Run """C:\Users\kjets\capture_trader\run_alpaca_bot.bat""", 0, False
