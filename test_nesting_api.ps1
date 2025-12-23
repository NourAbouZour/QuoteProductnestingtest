# PowerShell script to test Nesting Center API
# Usage: .\test_nesting_api.ps1

$baseUrl = "http://localhost:5000"

Write-Host "Testing Nesting Center API" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Test 1: Status endpoint
Write-Host "`n1. Testing status endpoint..." -ForegroundColor Yellow
try {
    $statusResponse = Invoke-RestMethod -Uri "$baseUrl/api/nesting-center/status" -Method Get
    Write-Host "   ✓ Status check successful" -ForegroundColor Green
    Write-Host "   Service: $($statusResponse.service)" -ForegroundColor Gray
    Write-Host "   Available: $($statusResponse.available)" -ForegroundColor Gray
} catch {
    Write-Host "   ✗ Status check failed: $_" -ForegroundColor Red
    exit 1
}

# Test 2: Optimize endpoint
Write-Host "`n2. Testing optimize endpoint..." -ForegroundColor Yellow
Write-Host "   (This may take 30-60 seconds)" -ForegroundColor Gray

$testData = @{
    parts = @(
        @{
            id = "test_part_1"
            length_mm = 200
            width_mm = 100
            quantity = 5
        }
    )
    boards = @(
        @{
            id = "test_board_1"
            width_mm = 3000
            height_mm = 1500
            quantity = 1
        }
    )
    settings = @{
        gap_mm = 5.0
        margin_mm = 5.0
        rotation = "Fixed90"
        timeout = 60
    }
} | ConvertTo-Json -Depth 10

try {
    Write-Host "   Sending request..." -ForegroundColor Gray
    $optimizeResponse = Invoke-RestMethod -Uri "$baseUrl/api/nesting-center/optimize" -Method Post -ContentType "application/json" -Body $testData -TimeoutSec 120
    
    if ($optimizeResponse.success) {
        Write-Host "   ✓ Nesting computation successful!" -ForegroundColor Green
        Write-Host "   Parts nested: $($optimizeResponse.total_parts_nested)" -ForegroundColor Gray
        Write-Host "   Plates used: $($optimizeResponse.total_plates_used)" -ForegroundColor Gray
        Write-Host "   SVG layouts: $($optimizeResponse.svg_layouts.Count)" -ForegroundColor Gray
        
        # Save result to file
        $resultFile = "test_nesting_result.json"
        $optimizeResponse | ConvertTo-Json -Depth 10 | Out-File -FilePath $resultFile -Encoding UTF8
        Write-Host "`n   Full result saved to: $resultFile" -ForegroundColor Cyan
        
        Write-Host "`n🎉 All tests passed! Nesting is working correctly." -ForegroundColor Green
    } else {
        Write-Host "   ✗ Nesting failed: $($optimizeResponse.error)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "   ✗ Request failed: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "   Response: $responseBody" -ForegroundColor Red
    }
    exit 1
}
