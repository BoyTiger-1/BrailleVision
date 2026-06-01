# Start BrailleVision backend + frontend (two windows)
$root = $PSScriptRoot

Write-Host "Starting BrailleVision backend on :8000 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; if (Test-Path .venv) { .\.venv\Scripts\Activate.ps1 }; pip install -q -r requirements.txt; python main.py"

Start-Sleep -Seconds 2

Write-Host "Starting frontend on :5173 ..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm install; npm run dev"

Write-Host "Open http://localhost:5173 when ready."
