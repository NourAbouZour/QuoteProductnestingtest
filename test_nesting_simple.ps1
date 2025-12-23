# Simple PowerShell test for Nesting Center API
$body = @{
    parts = @(@{id="test1"; length_mm=200; width_mm=100; quantity=5})
    boards = @(@{id="board1"; width_mm=3000; height_mm=1500; quantity=1})
} | ConvertTo-Json

Write-Host "Testing nesting API..." -ForegroundColor Yellow
$result = Invoke-RestMethod -Uri "http://localhost:5000/api/nesting-center/optimize" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 120
Write-Host "Result: $($result | ConvertTo-Json -Depth 5)" -ForegroundColor Green
