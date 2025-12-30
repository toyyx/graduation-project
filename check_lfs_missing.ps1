Write-Host "=== 检查 LFS 遗漏文件 ===" -ForegroundColor Cyan

# 1. 获取 LFS 已管理的文件
$lfs_files = @{}
git lfs ls-files | ForEach-Object {
    $parts = $_ -split '\s+'
    if ($parts.Count -ge 3) {
        $lfs_files[$parts[2]] = $true
    }
}

Write-Host "LFS 已管理文件数: $($lfs_files.Count)" -ForegroundColor Yellow

# 2. 查找所有大文件
Write-Host "`n查找大于 100MB 的文件..." -ForegroundColor Yellow
$large_files = Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt 100MB }

Write-Host "发现 $($large_files.Count) 个大文件 (>100MB)" -ForegroundColor Yellow

# 3. 找出遗漏的文件
$missing_files = @()
foreach ($file in $large_files) {
    $rel_path = $file.FullName.Replace("$(Get-Location)\", "")
    if (-not $lfs_files.ContainsKey($rel_path)) {
        $size_mb = [math]::Round($file.Length/1MB, 2)
        $missing_files += [PSCustomObject]@{
            Path = $rel_path
            SizeMB = $size_mb
            Extension = $file.Extension
        }
    }
}

# 4. 显示结果
if ($missing_files.Count -gt 0) {
    Write-Host "`n❌ 发现 $($missing_files.Count) 个被 LFS 遗漏的文件：" -ForegroundColor Red
    $missing_files | Format-Table SizeMB, Extension, Path -AutoSize
    
    # 按扩展名统计
    Write-Host "`n📊 按扩展名统计：" -ForegroundColor Cyan
    $missing_files | Group-Object Extension | ForEach-Object {
        $total_mb = ($_.Group | Measure-Object -Property SizeMB -Sum).Sum
        Write-Host "  $($_.Name): $($_.Count) 个文件, 共 $total_mb MB"
    }
} else {
    Write-Host "`n✅ 所有大文件均已由 LFS 管理" -ForegroundColor Green
}

# 5. 检查 LFS 规则匹配
Write-Host "`n🔧 检查 LFS 规则匹配：" -ForegroundColor Cyan
if ($missing_files.Count -gt 0) {
    foreach ($file in $missing_files) {
        $attr = git check-attr filter -- $file.Path 2>$null
        Write-Host "  $($file.Path): $attr"
    }
}
EOF
