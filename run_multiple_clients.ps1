# run_multiple_clients.ps1
Write-Host "Starting 3 Federated Clients..."
Start-Process powershell -ArgumentList "-NoExit -Command `".\venv\Scripts\python federated\client.py --name BankA`""
Start-Process powershell -ArgumentList "-NoExit -Command `".\venv\Scripts\python federated\client.py --name BankB`""
Start-Process powershell -ArgumentList "-NoExit -Command `".\venv\Scripts\python federated\client.py --name BankC`""
Write-Host "Clients launched in separate windows!"
