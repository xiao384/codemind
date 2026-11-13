<#
.SYNOPSIS
    CodeMind 服务器端一键训练脚本（Windows / PowerShell）。

.DESCRIPTION
    按「准备数据 -> 基准测试 -> 长跑训练 -> 抽样生成」的顺序执行，
    并在训练崩溃（尤其是 Windows 上的内存 native 崩溃）时自动降 batch 重启续训。

    为什么需要自动重启：Windows 内存耗尽时进程以 0xC0000005 / 0xC0000142 退出，
    **不会打印 Python 回溯**，一晚上的训练可能白费。
    本脚本把这种崩溃识别出来，降低 batch 后续训（latest.pt 每 ckpt-interval 步落盘）。

.EXAMPLE
    # 1) 先看机器能跑多快（几分钟）
    .\tools\run_server.ps1 -Mode bench

    # 2) 准备数据（100MB-1GB 代码）
    .\tools\run_server.ps1 -Mode prepare -Source D:\corpora\github,D:\corpora\internal -Vocab 16384

    # 3) 长跑训练（自动续训、崩溃自愈）
    .\tools\run_server.ps1 -Mode train -Run runs\small -Preset small -TargetSteps 20000

    # 4) 抽样生成看效果
    .\tools\run_server.ps1 -Mode sample -Run runs\small

.NOTES
    环境要求：Python 3.10+，已安装 torch + numpy（CPU 版即可）。
    推荐先 `uv pip install torch numpy --index-url https://download.pytorch.org/whl/cpu`
#>

[CmdletBinding()]
param(
    [ValidateSet("bench", "prepare", "train", "sample", "all")]
    [string]$Mode = "bench",

    # ---- 数据 ----
    [string[]]$Source = @(),
    [string]$DataName = "corpus",
    [string]$DataDir = "data",
    [int]$Vocab = 16384,
    [string]$Preset = "small",
    [long]$ShardTokens = 20000000,
    [int]$TokenizerChars = 50000000,

    # ---- 训练 ----
    [string]$Run = "",
    [int]$TargetSteps = 0,
    [int]$BatchSize = 0,
    [int]$GradAccum = 1,
    [int]$Threads = 0,
    [string]$Amp = "off",
    [int]$CkptInterval = 500,
    [int]$EvalInterval = 1000,
    [double]$Lr = 0,
    [int]$CheckpointEverySteps = 0,   # 每多少步做一次崩溃自愈检查（默认跟随 ckpt-interval）

    [int]$MaxRestarts = 20,
    [switch]$NoBench
)

$ErrorActionPreference = "Stop"

# $PSScriptRoot 在部分调用方式下会是空字符串，一旦为空 Split-Path 直接抛错。
# 用 $MyInvocation 兜底，否则脚本还没开始就崩了。
$script:ScriptDir = $PSScriptRoot
if (-not $script:ScriptDir) {
    $script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
}
$script:RepoRoot = Split-Path -Parent $script:ScriptDir
Set-Location $script:RepoRoot

# ----------------------------------------------------------------------------- 工具

function Write-Head([string]$text) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkCyan
}

function Get-Python {
    $candidates = @(
        (Join-Path $script:RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $script:RepoRoot ".venv/bin/python")
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return "python"
}

function Get-SystemMemoryGB {
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        return [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    } catch { return 0 }
}

function Resolve-AutoBatch([int]$memGB, [string]$preset) {
    # 依据物理内存给出保守的 batch 初值（宁可小，崩溃重启代价高）
    if ($memGB -le 0) { return 4 }
    $table = @{
        nano   = @{ 8 = 32; 16 = 64; 32 = 96; 64 = 128 }
        tiny   = @{ 8 = 8;  16 = 24; 32 = 48; 64 = 64 }
        small  = @{ 8 = 2;  16 = 4;  32 = 8;  64 = 16 }
        medium = @{ 8 = 1;  16 = 2;  32 = 4;  64 = 8 }
    }
    if (-not $table.ContainsKey($preset)) { return 4 }
    $tiers = $table[$preset].Keys | Sort-Object
    foreach ($tier in $tiers) { if ($memGB -le $tier) { return $table[$preset][$tier] } }
    return $table[$preset][($tiers | Select-Object -Last 1)]
}

function Invoke-CodeMind {
    param([string[]]$Arguments, [switch]$AllowFail)
    $py = Get-Python
    $env:PYTHONIOENCODING = "utf-8"
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
    # 关键：PowerShell 会把原生命令的 stdout 当成"返回值"收集走，
    # 直接 `return $code` 会让调用方看不到任何输出（整个脚本像是没干活）。
    # 用 Out-Host 立即把子进程输出写到控制台，只让退出码成为返回值。
    & $py -u -m codemind @Arguments 2>&1 | Out-Host
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFail) {
        throw "codemind $($Arguments -join ' ') 失败，退出码 $code"
    }
    return $code
}

function Test-LooksLikeOom([int]$exitCode) {
    # Windows 内存枯竭的典型 native 崩溃码（无 Python 回溯）
    #   0xC0000005 = -1073741819  访问冲突
    #   0xC0000142 = -1073741502  DLL 初始化失败
    #   0xC0000409 = -1073740791  栈缓冲溢出 / 快速失败
    return @(-1073741819, -1073741502, -1073740791) -contains $exitCode
}

# ----------------------------------------------------------------------------- 模式

function Invoke-Bench {
    Write-Head "步骤 1/4：性能基准（决定能训多大的模型）"
    $args = @("bench", "--presets", "nano", "tiny", "small")
    if ($Threads -gt 0) { $args += @("--threads", "$Threads") }
    $args += @("--batch-size", "$(if ($BatchSize -gt 0) { $BatchSize } else { 8 })")
    $args += @("--steps", "8")
    Invoke-CodeMind -Arguments $args -AllowFail | Out-Null
    Write-Host ""
    Write-Host "解读方式：" -ForegroundColor Yellow
    Write-Host "  * 挑「训练 tok/s」最快的那一行，把它对应的线程数固定下来（--Threads）。"
    Write-Host "  * 用 Chinchilla 那一节判断：哪个预设能在你能接受的时长内跑完。"
    Write-Host "  * 纯 CPU 的实用上限约 100M 参数；再大就不是慢，而是跑不完。"
}

function Invoke-Prepare {
    Write-Head "步骤 2/4：准备数据（采集 -> 训练分词器 -> 切分片）"
    if ($Source.Count -eq 0) {
        throw "请用 -Source 指定代码目录，例如 -Source D:\corpora\github,D:\corpora\internal"
    }
    $missing = $Source | Where-Object { -not (Test-Path $_) }
    if ($missing) { throw "以下目录不存在: $($missing -join ', ')" }

    foreach ($dir in $Source) {
        $sizeMB = [math]::Round((Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Host ("  源目录 {0}  约 {1} MB" -f $dir, $sizeMB)
    }

    $args = @(
        "prepare",
        "--name", $DataName,
        "--out", $DataDir,
        "--vocab", "$Vocab",
        "--preset", $Preset,
        "--shard-tokens", "$ShardTokens",
        "--tokenizer-chars", "$TokenizerChars"
    )
    foreach ($s in $Source) { $args += @("--source", $s) }
    Invoke-CodeMind -Arguments $args
}

function Invoke-Train {
    Write-Head "步骤 3/4：训练（崩溃自愈 + 自动续训）"

    $memGB = Get-SystemMemoryGB
    $batch = if ($BatchSize -gt 0) { $BatchSize } else { Resolve-AutoBatch $memGB $Preset }
    $runDir = if ($Run) { $Run } else { "runs\$DataName" }
    $ckpt = if ($CheckpointEverySteps -gt 0) { $CheckpointEverySteps } else { $CkptInterval }
    $steps = if ($TargetSteps -gt 0) { $TargetSteps } else { 20000 }

    Write-Host ("  物理内存 {0} GB -> batch 初值 {1}（grad-accum {2}，等效 {3}）" -f `
        $memGB, $batch, $GradAccum, ($batch * $GradAccum))
    Write-Host ("  目标 {0} 步，每 {1} 步落一次盘" -f $steps, $ckpt)

    $attempt = 0
    $current = $batch
    while ($attempt -lt $MaxRestarts) {
        $attempt++
        $args = @(
            "train",
            "--run", $runDir,
            "--data", (Join-Path $DataDir $DataName),
            "--preset", $Preset,
            "--steps", "$steps",
            "--batch-size", "$current",
            "--grad-accum", "$GradAccum",
            "--ckpt-interval", "$ckpt",
            "--eval-interval", "$EvalInterval",
            "--amp", $Amp
        )
        if ($Threads -gt 0) { $args += @("--threads", "$Threads") }
        if ($Lr -gt 0) { $args += @("--lr", "$Lr") }

        Write-Host ""
        Write-Host ("--- 第 {0} 次尝试 | batch={1} ---" -f $attempt, $current) -ForegroundColor Green
        $py = Get-Python
        $env:PYTHONIOENCODING = "utf-8"
        try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
        & $py -u -m codemind @args
        $code = $LASTEXITCODE

        if ($code -eq 0) {
            Write-Host ""
            Write-Host "训练正常结束。" -ForegroundColor Green
            return
        }

        if (Test-LooksLikeOom $code) {
            Write-Host ""
            Write-Host ("检测到内存类崩溃（退出码 0x{0:X}）。" -f $code) -ForegroundColor Yellow
            if ($current -le 1 -and $GradAccum -lt 16) {
                $GradAccum *= 2
                Write-Host "batch 已到 1，改为提高梯度累积到 $GradAccum 以维持等效 batch。" -ForegroundColor Yellow
            } elseif ($current -gt 1) {
                $current = [math]::Max(1, [int]($current / 2))
                Write-Host "batch 减半为 $current，从 latest.pt 续训。" -ForegroundColor Yellow
            } else {
                throw "batch=1 仍内存不足，请换更小的 --Preset（如 tiny）或增大 --ShardTokens 之外的其它内存占用。"
            }
            Start-Sleep -Seconds 5
            continue
        }

        Write-Host "训练以退出码 $code 结束（非内存问题），不再自动重启。" -ForegroundColor Red
        Write-Host "若为中断（Ctrl-C），直接重跑本命令即可从 latest.pt 续训。" -ForegroundColor Yellow
        return
    }
    Write-Host "已达到最大重启次数 $MaxRestarts。" -ForegroundColor Red
}

function Invoke-Sample {
    Write-Head "步骤 4/4：抽样生成（看真实效果）"
    $runDir = if ($Run) { $Run } else { "runs\$DataName" }
    $probes = @(
        @("续写函数", "def add(a, b):`n"),
        @("续写类", "class DataLoader:`n"),
        @("类型注解", "from typing import Optional`n`ndef process(items: list[str]) -> Optional[int]:`n")
    )
    foreach ($p in $probes) {
        Write-Host ""
        Write-Host ("--- {0} ---" -f $p[0]) -ForegroundColor Cyan
        Write-Host ("提示: {0}" -f $p[1].Replace("`n", "\n"))
        Invoke-CodeMind -Arguments @(
            "gen", "--run", $runDir, "--prompt", $p[1],
            "--max-tokens", "64", "--temperature", "0.4", "--greedy", "--no-stream"
        ) -AllowFail | Out-Null
    }
    Write-Host ""
    Write-Host ("也可以看检查点信息:  python -m codemind info --run {0}" -f $runDir) -ForegroundColor Yellow
}

# ----------------------------------------------------------------------------- 主流程

$memGB = Get-SystemMemoryGB
Write-Head "CodeMind 服务器训练脚本"
Write-Host ("工作目录  : {0}" -f $script:RepoRoot)
Write-Host ("Python    : {0}" -f (Get-Python))
if ($memGB -gt 0) { Write-Host ("物理内存  : {0} GB" -f $memGB) }

switch ($Mode) {
    "bench"   { Invoke-Bench }
    "prepare" { Invoke-Prepare }
    "train"   { Invoke-Train }
    "sample"  { Invoke-Sample }
    "all"     {
        if (-not $NoBench) { Invoke-Bench }
        Invoke-Prepare
        Invoke-Train
        Invoke-Sample
    }
}

Write-Host ""
Write-Host "完成。" -ForegroundColor Green
