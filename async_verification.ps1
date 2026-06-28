#Requires -Version 5.1
<#
.SYNOPSIS
    Verifies async is working and shows clean text output.
#>

$BaseUrl = "http://localhost:8000"
$ConcurrentRequests = 3

$Queries = @(
    "What is the current repo rate?",
    "Calculate the percentage increase from 4.0 to 6.5",
    "What is GDP growth outlook?"
)

function Send-ChatRequest {
    param([string]$Query, [int]$Idx)
    $body = @{
        message = $Query
        conversation_id = "async-test-$Idx"
    } | ConvertTo-Json -Depth 3

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/api/v1/chat" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 60
        $sw.Stop()
        return [PSCustomObject]@{
            Idx            = $Idx
            Status         = 200
            ElapsedSec     = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            ResponsePreview= if ($response.response) { $response.response.Substring(0, [Math]::Min(100, $response.response.Length)) } else { "N/A" }
            ConversationId = $response.conversation_id
        }
    } catch {
        $sw.Stop()
        return [PSCustomObject]@{
            Idx        = $Idx
            Status     = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { 0 }
            ElapsedSec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            Error      = $_.Exception.Message
        }
    }
}

Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "ASYNC VERIFICATION" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green

# Health
Write-Host "`n[1] Health check..." -ForegroundColor Cyan
$health = Invoke-RestMethod -Uri "$BaseUrl/api/v1/health"
Write-Host "    Status: $($health.status) | Store: $($health.store.mode)"

# Single request
Write-Host "`n[2] Single request baseline..." -ForegroundColor Cyan
$single = Send-ChatRequest -Query $Queries[0] -Idx 0
Write-Host "    Time: $($single.ElapsedSec)s"
Write-Host "    Response: $($single.ResponsePreview)"

# Concurrent requests
Write-Host "`n[3] Firing $ConcurrentRequests concurrent requests..." -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$jobs = for ($i = 0; $i -lt $ConcurrentRequests; $i++) {
    Start-Job -ScriptBlock {
        param($Url, $Query, $Idx)
        $body = @{ message = $Query; conversation_id = "async-test-$Idx" } | ConvertTo-Json -Depth 3
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $r = Invoke-RestMethod -Uri "$Url/api/v1/chat" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 60
            $timer.Stop()
            [PSCustomObject]@{
                Idx = $Idx
                Status = 200
                ElapsedSec = [math]::Round($timer.Elapsed.TotalSeconds, 2)
                ResponsePreview = if ($r.response) { $r.response.Substring(0, [Math]::Min(100, $r.response.Length)) } else { "N/A" }
            }
        } catch {
            $timer.Stop()
            [PSCustomObject]@{
                Idx = $Idx
                Status = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { 0 }
                ElapsedSec = [math]::Round($timer.Elapsed.TotalSeconds, 2)
                Error = $_.Exception.Message
            }
        }
    } -ArgumentList $BaseUrl, $Queries[$i], $i
}

$results = $jobs | ForEach-Object { $_ | Wait-Job | Receive-Job; Remove-Job $_ }
$sw.Stop()
$totalTime = [math]::Round($sw.Elapsed.TotalSeconds, 2)

foreach ($r in $results) {
    if ($r.Error) {
        Write-Host "    Request $($r.Idx): ERROR $($r.Status) in $($r.ElapsedSec)s" -ForegroundColor Red
    } else {
        Write-Host "    Request $($r.Idx): OK in $($r.ElapsedSec)s | $($r.ResponsePreview)"
    }
}

# Analysis
$ratio = if ($single.ElapsedSec -gt 0) { [math]::Round($totalTime / $single.ElapsedSec, 2) } else { 0 }
Write-Host "`nTotal wall time: ${totalTime}s | Concurrency ratio: ${ratio}x" -ForegroundColor Yellow

if ($ratio -lt 1.5) {
    Write-Host "PASS: Requests are CONCURRENT. Event loop is NOT blocked." -ForegroundColor Green
} elseif ($ratio -lt 2.5) {
    Write-Host "WARNING: Partial concurrency." -ForegroundColor Yellow
} else {
    Write-Host "FAIL: Requests are SEQUENTIAL. Event loop IS blocked." -ForegroundColor Red
}

# Redis persistence
Write-Host "`n[4] Redis persistence check..." -ForegroundColor Cyan
$trace = Invoke-RestMethod -Uri "$BaseUrl/api/v1/trace/async-test-0"
Write-Host "    Found: $($trace.conversation_id) | Turns: $($trace.turn_count) | Store: $($trace.store)"