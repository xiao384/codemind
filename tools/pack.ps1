<#
.SYNOPSIS
    把 CodeMind 打成一个干净的压缩包，便于迁移到服务器。

.DESCRIPTION
    只打包源码与配置，**排除**虚拟环境、训练产物、语料分片与各种缓存。
    原因：.venv 不可跨机复制（里面有硬编码的绝对路径和平台相关的二进制），
    服务器上重新安装依赖只要一两分钟，却能让包从几百 MB 缩到几十 KB。

.EXAMPLE
    .\tools\pack.ps1
    .\tools\pack.ps1 -Out D:\codemind-transfer.zip
#>

[CmdletBinding()]
param(
    [string]$Out = "",
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot 在部分调用方式（如某些 powershell -File 组合）下会是空字符串，
# 一旦为空，Split-Path 会直接抛错、脚本第一行就崩。这里用 $MyInvocation 兜底。
if (-not $RepoRoot) {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
    }
    $RepoRoot = Split-Path -Parent $scriptDir
}
Set-Location $RepoRoot

if (-not $Out) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmm"
    $Out = Join-Path (Split-Path -Parent $RepoRoot) "codemind-$stamp.zip"
}

# 需要打包的内容（白名单：只带源码与文档，避免把产物/缓存带过去）
$include = @(
    "codemind",
    "tests",
    "tools",
    "README.md",
    "MIGRATION.md",
    "pyproject.toml",
    "pytest.ini",
    ".gitignore"
)

# 额外排除（白名单之内仍要剔除的）
$excludeDirs = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".venv")

Write-Host "打包 CodeMind" -ForegroundColor Cyan
Write-Host ("  源目录: {0}" -f $RepoRoot)
Write-Host ("  目标  : {0}" -f $Out)

$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("codemind-pack-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $staging -Force | Out-Null

try {
    $copied = 0
    foreach ($item in $include) {
        $src = Join-Path $RepoRoot $item
        if (-not (Test-Path $src)) {
            Write-Host ("  跳过（不存在）: {0}" -f $item) -ForegroundColor DarkYellow
            continue
        }
        if (Test-Path $src -PathType Container) {
            Get-ChildItem -Path $src -Recurse -File | ForEach-Object {
                $relative = $_.FullName.Substring($RepoRoot.Length).TrimStart('\', '/')
                $skip = $false
                foreach ($bad in $excludeDirs) {
                    if ($relative -split '[\\/]' -contains $bad) { $skip = $true; break }
                }
                if ($skip) { return }
                $dest = Join-Path $staging $relative
                $destDir = Split-Path -Parent $dest
                if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
                Copy-Item $_.FullName $dest -Force
                $script:copied++
            }
        } else {
            Copy-Item $src (Join-Path $staging $item) -Force
            $script:copied++
        }
    }

    # 附一份服务器上快速上手的说明
    $quickstart = @'
# 服务器上 3 步跑起来

## 0) 环境（只需一次）
```powershell
# 如果没装 uv：pip install uv
uv venv
uv pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest
```

## 1) 自检 + 基准（几分钟，先摸清这台机器能跑多快）
```powershell
python -m codemind doctor
python -m codemind selftest          # 确认全链路可用
python -m codemind bench --presets nano tiny small
```

## 2) 准备数据
```powershell
python -m codemind prepare --source D:\corpora\github --source D:\corpora\internal `
    --name corpus --vocab 16384 --preset small --tokenizer-chars 50000000
```
`prepare` 会输出「这份数据够训多大的模型」，照着它选预设。

## 3) 长跑训练（崩溃自愈 + 自动续训）
```powershell
.\tools\run_server.ps1 -Mode train -Run runs\small -Preset small -TargetSteps 20000
```

## 4) 看效果
```powershell
python -m codemind gen --run runs\small --prompt "def add(a, b):" --max-tokens 64
python -m codemind fim --run runs\small -f some_file.py --line 40
```

## Windows 注意事项
- 脚本执行策略：`powershell -ExecutionPolicy Bypass -File .\tools\run_server.ps1 ...`
- 中文乱码：`$env:PYTHONIOENCODING="utf-8"` 与 `[Console]::OutputEncoding=[Text.Encoding]::UTF8`
- 内存不足会**静默崩溃**（退出码 0xC0000005，无回溯）：降 `--batch-size`、加 `--grad-accum`、限制 `--threads`
'@
    Set-Content -Path (Join-Path $staging "SERVER_QUICKSTART.md") -Value $quickstart -Encoding UTF8

    if (Test-Path $Out) { Remove-Item $Out -Force }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $Out -CompressionLevel Optimal

    $sizeKB = [math]::Round((Get-Item $Out).Length / 1KB, 1)
    Write-Host ""
    Write-Host ("完成：{0} 个文件 -> {1} ({2} KB)" -f $script:copied, $Out, $sizeKB) -ForegroundColor Green
    Write-Host ""
    Write-Host "服务器上：" -ForegroundColor Yellow
    Write-Host ("  Expand-Archive -Path '{0}' -DestinationPath D:\codemind" -f (Split-Path -Leaf $Out))
    Write-Host "  cd D:\codemind; Get-Content SERVER_QUICKSTART.md"
}
finally {
    Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
}
