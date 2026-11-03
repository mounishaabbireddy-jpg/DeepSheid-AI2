Add-Type -AssemblyName System.Drawing

$dirs = @(
  "hackathon_dataset/train/real",
  "hackathon_dataset/train/fake",
  "hackathon_dataset/test/images"
)

foreach ($d in $dirs) {
  if (!(Test-Path $d)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
  }
}

function Create-FaceImage($path, $isFake, $seed) {
  $bmp = New-Object System.Drawing.Bitmap 224, 224
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

  # Background
  $bgColors = @(
    [System.Drawing.Color]::FromArgb(35, 45, 60),
    [System.Drawing.Color]::FromArgb(45, 55, 72),
    [System.Drawing.Color]::FromArgb(25, 30, 45),
    [System.Drawing.Color]::FromArgb(50, 40, 55)
  )
  $bgBrush = New-Object System.Drawing.SolidBrush ($bgColors[$seed % $bgColors.Length])
  $g.FillRectangle($bgBrush, 0, 0, 224, 224)

  # Face Oval
  $faceBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(235, 195, 165))
  $g.FillEllipse($faceBrush, 45, 35, 134, 155)

  # Hair
  $hairBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(40, 30, 20))
  $g.FillEllipse($hairBrush, 40, 20, 144, 55)

  # Eyes
  $eyeBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(40, 30, 25))
  $whiteBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::White)
  
  $g.FillEllipse($whiteBrush, 70, 85, 24, 14)
  $g.FillEllipse($whiteBrush, 130, 85, 24, 14)
  $g.FillEllipse($eyeBrush, 77, 87, 10, 10)
  $g.FillEllipse($eyeBrush, 137, 87, 10, 10)

  # Nose
  $nosePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(190, 140, 110), 2)
  $g.DrawLine($nosePen, 112, 95, 108, 125)
  $g.DrawLine($nosePen, 108, 125, 116, 125)

  # Mouth
  $mouthPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(180, 80, 80), 3)
  $g.DrawArc($mouthPen, 90, 140, 44, 20, 10, 160)

  # If Fake, introduce visible synthetic artifacts:
  if ($isFake) {
    # Frequency checkerboard artifact / boundary seam
    $seamPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(160, 255, 60, 120), 2)
    $seamPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
    $g.DrawEllipse($seamPen, 43, 33, 138, 159)

    # Inconsistent specular reflection
    $glitchBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(200, 0, 240, 255))
    $g.FillRectangle($glitchBrush, 75, 85, 6, 6)

    # High frequency grid line
    $gridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(40, 255, 255, 255), 1)
    for ($y = 0; $y -lt 224; $y += 16) {
      $g.DrawLine($gridPen, 0, $y, 224, $y)
    }
  }

  $codec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' }
  $params = New-Object System.Drawing.Imaging.EncoderParameters 1
  $params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter ([System.Drawing.Imaging.Encoder]::Quality, [long]90)

  $bmp.Save($path, $codec, $params)
  $g.Dispose()
  $bmp.Dispose()
}

# Generate 8 real images
for ($i = 1; $i -le 8; $i++) {
  $name = "real_face_{0:D3}.jpg" -f $i
  Create-FaceImage "hackathon_dataset/train/real/$name" $false $i
}

# Generate 8 fake images
for ($i = 1; $i -le 8; $i++) {
  $name = "fake_face_{0:D3}.jpg" -f $i
  Create-FaceImage "hackathon_dataset/train/fake/$name" $true ($i + 50)
}

# Generate test images with UUIDs matching sample_submission
$testIds = @(
  "3f9a2c1d-8b4e-4f32-9c1a-2e8d7a1b0c3f",
  "a8b4e7f2-1d3c-4e5a-b7f9-8a2c1d3e4f5a",
  "c7d8e9f0-2a3b-4c5d-6e7f-8a9b0c1d2e3f",
  "e1f2a3b4-5c6d-7e8f-9a0b-1c2d3e4f5a6b",
  "d4e5f6a7-b8c9-0d1e-2f3a-4b5c6d7e8f9a",
  "b9c0d1e2-f3a4-5b6c-7d8e-9f0a1b2c3d4e",
  "f5a6b7c8-d9e0-1a2b-3c4d-5e6f7a8b9c0d",
  "0a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
)

$idx = 0
foreach ($tid in $testIds) {
  $isF = ($idx % 2 -eq 0)
  Create-FaceImage "hackathon_dataset/test/images/$tid.jpg" $isF ($idx * 7)
  $idx++
}

Write-Host "Sample dataset generated successfully in hackathon_dataset/"

