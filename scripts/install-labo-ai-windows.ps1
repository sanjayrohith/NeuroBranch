$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repository = 'Complexity-ML/labo-ai'
$asset = 'LABO-AI-Setup-x64-helper.exe'
$latestUrl = "https://github.com/$repository/releases/latest/download"
$profileRoot = if ($env:APPDATA) { $env:APPDATA } else { Join-Path $env:USERPROFILE 'AppData\Roaming' }
$installDirectory = Join-Path $profileRoot 'LABO AI\installer'
$installPath = Join-Path $installDirectory 'labo-ai-setup.exe'
$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("labo-ai-setup-" + [guid]::NewGuid().ToString('N'))

function Get-LaboAsset {
  param([string]$Uri, [string]$OutFile)
  $lastError = $null
  foreach ($attempt in 1..4) {
    try {
      Invoke-WebRequest -UseBasicParsing -Headers @{ 'User-Agent' = 'LABO-AI-Setup' } -Uri $Uri -OutFile $OutFile
      return
    } catch {
      $lastError = $_
      if ($attempt -lt 4) { Start-Sleep -Seconds (2 * $attempt) }
    }
  }
  throw "Unable to download $Uri after 4 attempts: $lastError"
}

try {
  New-Item -ItemType Directory -Force -Path $temporaryDirectory | Out-Null
  $downloadedHelper = Join-Path $temporaryDirectory $asset
  $downloadedDigest = "$downloadedHelper.sha256"

  Write-Host 'Downloading the latest verified LABO AI Setup...'
  Get-LaboAsset -Uri "$latestUrl/$asset" -OutFile $downloadedHelper
  Get-LaboAsset -Uri "$latestUrl/$asset.sha256" -OutFile $downloadedDigest

  $expectedDigest = ((Get-Content $downloadedDigest -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
  $actualDigest = (Get-FileHash $downloadedHelper -Algorithm SHA256).Hash.ToLowerInvariant()
  if ([string]::IsNullOrWhiteSpace($expectedDigest) -or $actualDigest -ne $expectedDigest) {
    throw 'LABO AI Setup checksum verification failed. Nothing was installed.'
  }

  New-Item -ItemType Directory -Force -Path $installDirectory | Out-Null
  Copy-Item -Force $downloadedHelper "$installPath.next"
  Move-Item -Force "$installPath.next" $installPath

  Write-Host 'Verified. Opening LABO AI Setup...'
  $process = Start-Process -FilePath $installPath -ArgumentList '--auto-install' -PassThru
  Start-Sleep -Milliseconds 800
  if ($process.HasExited -and $process.ExitCode -ne 0) {
    throw "LABO AI Setup exited immediately with code $($process.ExitCode)."
  }
} finally {
  if (Test-Path $temporaryDirectory) {
    Remove-Item -Recurse -Force $temporaryDirectory
  }
}
