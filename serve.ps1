$port = 8080
$prefix = "http://localhost:$port/"
$baseDir = $PSScriptRoot

Write-Host "========================================================"
Write-Host " DEEPSHIELD-X — Intelligence Beyond Appearance"
Write-Host " Enterprise AI Forensics Platform Web Server"
Write-Host " Running at: $prefix"
Write-Host " Press Ctrl+C in this terminal window to stop."
Write-Host "========================================================"

# Try HttpListener first; fall back to TcpListener if ACL is restricted
$useHttpListener = $false
try {
    $listener = New-Object System.Net.HttpListener
    $listener.Prefixes.Add($prefix)
    $listener.Start()
    $useHttpListener = $true
} catch {
    $useHttpListener = $false
}

# Launch browser if interactive environment
try {
    Start-Process $prefix -ErrorAction SilentlyContinue
} catch {}

function Get-ContentType($filePath) {
    $ext = [System.IO.Path]::GetExtension($filePath).ToLower()
    switch ($ext) {
        ".html"  { return "text/html; charset=utf-8" }
        ".css"   { return "text/css; charset=utf-8" }
        ".js"    { return "application/javascript; charset=utf-8" }
        ".json"  { return "application/json; charset=utf-8" }
        ".jpg"   { return "image/jpeg" }
        ".jpeg"  { return "image/jpeg" }
        ".png"   { return "image/png" }
        ".webp"  { return "image/webp" }
        ".svg"   { return "image/svg+xml" }
        ".csv"   { return "text/csv; charset=utf-8" }
        ".md"    { return "text/markdown; charset=utf-8" }
        ".woff2" { return "font/woff2" }
        default  { return "application/octet-stream" }
    }
}

if ($useHttpListener) {
    while ($listener.IsListening) {
        try {
            $context = $listener.GetContext()
            $request = $context.Request
            $response = $context.Response

            $rawPath = $request.Url.LocalPath

            if ($rawPath -eq "/api/health") {
                $response.StatusCode = 200
                $response.ContentType = "application/json; charset=utf-8"
                $json = '{"status":"ok","model":"DeepShieldXNet-v2.4-dualstream","engine":"active"}'
                $buffer = [System.Text.Encoding]::UTF8.GetBytes($json)
                $response.ContentLength64 = $buffer.Length
                $response.OutputStream.Write($buffer, 0, $buffer.Length)
                $response.OutputStream.Close()
                continue
            }

            $urlPath = $rawPath.TrimStart('/')
            if ([string]::IsNullOrEmpty($urlPath)) {
                $urlPath = "index.html"
            }

            $filePath = Join-Path $baseDir $urlPath
            $fullPath = [System.IO.Path]::GetFullPath($filePath)

            if (!$fullPath.StartsWith($baseDir, [System.StringComparison]::OrdinalIgnoreCase) -or !(Test-Path $fullPath)) {
                $response.StatusCode = 404
                $buffer = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found")
                $response.ContentLength64 = $buffer.Length
                $response.OutputStream.Write($buffer, 0, $buffer.Length)
                $response.OutputStream.Close()
                continue
            }

            $bytes = [System.IO.File]::ReadAllBytes($fullPath)
            $response.ContentType = Get-ContentType $fullPath
            $response.ContentLength64 = $bytes.Length
            $response.OutputStream.Write($bytes, 0, $bytes.Length)
            $response.OutputStream.Close()
        } catch {}
    }
} else {
    # TcpListener fallback
    $tcp = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $port)
    $tcp.Start()
    while ($true) {
        try {
            $client = $tcp.AcceptTcpClient()
            $stream = $client.GetStream()
            $reader = New-Object System.IO.StreamReader($stream)
            $firstLine = $reader.ReadLine()
            if ([string]::IsNullOrEmpty($firstLine)) {
                $client.Close()
                continue
            }

            $tokens = $firstLine.Split(" ")
            $rawPath = if ($tokens.Length -ge 2) { $tokens[1] } else { "/" }
            $cleanPath = $rawPath.Split("?")[0].TrimStart('/')
            if ([string]::IsNullOrEmpty($cleanPath)) { $cleanPath = "index.html" }

            if ($cleanPath -eq "api/health") {
                $body = [System.Text.Encoding]::UTF8.GetBytes('{"status":"ok","model":"DeepShieldXNet-v2.4-dualstream","engine":"active"}')
                $header = "HTTP/1.1 200 OK`r`nContent-Type: application/json; charset=utf-8`r`nContent-Length: $($body.Length)`r`nConnection: close`r`n`r`n"
                $hBytes = [System.Text.Encoding]::UTF8.GetBytes($header)
                $stream.Write($hBytes, 0, $hBytes.Length)
                $stream.Write($body, 0, $body.Length)
                $stream.Flush()
                $client.Close()
                continue
            }

            $fullPath = [System.IO.Path]::GetFullPath((Join-Path $baseDir $cleanPath))
            if (!$fullPath.StartsWith($baseDir, [System.StringComparison]::OrdinalIgnoreCase) -or !(Test-Path $fullPath)) {
                $body = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found")
                $header = "HTTP/1.1 404 Not Found`r`nContent-Type: text/plain`r`nContent-Length: $($body.Length)`r`nConnection: close`r`n`r`n"
                $hBytes = [System.Text.Encoding]::UTF8.GetBytes($header)
                $stream.Write($hBytes, 0, $hBytes.Length)
                $stream.Write($body, 0, $body.Length)
                $stream.Flush()
                $client.Close()
                continue
            }

            $bytes = [System.IO.File]::ReadAllBytes($fullPath)
            $contentType = Get-ContentType $fullPath
            $header = "HTTP/1.1 200 OK`r`nContent-Type: $contentType`r`nContent-Length: $($bytes.Length)`r`nConnection: close`r`n`r`n"
            $hBytes = [System.Text.Encoding]::UTF8.GetBytes($header)
            $stream.Write($hBytes, 0, $hBytes.Length)
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush()
            $client.Close()
        } catch {}
    }
}
