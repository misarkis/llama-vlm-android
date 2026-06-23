# Open VLM-Analyze Android project in Android Studio
$ANDROID_STUDIO = "C:\Program Files\Android\Android Studio\bin\studio64.exe"
$PROJECT_PATH = $PSScriptRoot

if (Test-Path $ANDROID_STUDIO) {
    Write-Host "Opening VLM-Analyze Android in Android Studio..."
    Start-Process $ANDROID_STUDIO -ArgumentList $PROJECT_PATH
} else {
    Write-Host "Android Studio not found at expected location."
    Write-Host "Please open Android Studio manually and import: $PROJECT_PATH"
}
