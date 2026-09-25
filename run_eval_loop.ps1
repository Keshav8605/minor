$TotalInferences = 40

while ($true) {
    # Check status
    $StatusOutput = python scripts/run_controlled_evaluation.py --status
    Write-Host $StatusOutput
    
    if ($StatusOutput -match "Completed:\s+40/40") {
        Write-Host "Evaluation fully completed!"
        break
    }
    
    # Run one inference
    Write-Host "Starting next inference..."
    $process = Start-Process -FilePath "python" -ArgumentList "scripts/run_controlled_evaluation.py", "--continue", "--run-one" -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-Host "Process crashed. Waiting 5 seconds before restarting..."
        Start-Sleep -Seconds 5
    }
}
