# PandaPush 历史复现命令

本文件保留完整实验的历史命令。此上传目录包含报告引用的数据、运行最终对照所需的文件以及部分检查点。早期历史续训命令需要相应的原始回放缓冲区。

**PandaPush — Reinforcement Learning for Contact-Rich Robotic Manipulation**

Franka Panda 在 MuJoCo 中用实体末端推动桌面方块。状态观测供 SB3 SAC 使用，稀疏奖励可配合官方 HER 重标。动作输出是三维末端位移增量，由六维 Jacobian IK 和七关节伺服执行。成功要求实际方块中心距目标小于 3 cm，连续至少 3 个控制步，并保持到回合结束。每回合最多 60 个控制步。

[开源源码调研](docs/open_source_review.md) · [中文算法笔记](docs/algorithm_notes.md) · [实验报告](docs/experiment_report.md)

## 安装和验证

在项目目录启动 PowerShell。已验证 Windows、Python 3.12.10、MuJoCo 3.4.0、Gymnasium 1.2.2、SB3 2.7.1 和 PyTorch 2.7.1 CPU。机器有 i7-13700H、约 16 GB 内存和 RTX 4060 Laptop GPU。这次使用 CPU 构建的 PyTorch，训练不需要窗口或 CUDA。

```powershell
uv venv --python 3.12 --seed .venv
uv pip sync --python .venv/Scripts/python.exe --index 'https://download.pytorch.org/whl/cpu' --index-strategy unsafe-best-match requirements-lock.txt
$py = '.\.venv\Scripts\python.exe'
& $py -m pip check
& $py -m pytest -q
```

## 演示

下面使用 50,000 步模型重放独立测试集中的成功场景。Viewer 运行一个回合后退出。[成功视频](results/test_50k/sac_dense_seed11/media/SAC_success.mp4) 和 [失败视频](results/test_50k/sac_dense_seed11/media/SAC_failure.mp4) 可直接打开。

```powershell
.\.venv\Scripts\python.exe -m panda_push.evaluate --checkpoint results/sac_dense/seed11_50k/final.zip --seed 61082 --episodes 1 --viewer
```

## 复现实验

所有训练输出路径必须是新目录，避免覆盖含模型和日志的旧运行。以下命令使用 `repro`。默认配置启用稀疏安全代价，见 [configs/default.json](configs/default.json)。下列原始对照使用冻结的 [configs/original.json](configs/original.json)。每个训练目录保存模型、回放缓冲、SB3 日志、Monitor 回合数据、定期开发评估和实际用时。续训使用同一目录里的 `final.zip`、`final_buffer.pkl` 和 `config.json`，从新回合开始。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m panda_push.evaluate --policy rule --layout fixed --seed 41000 --episodes 1 --output repro/baselines/rule_fixed
& $py -m panda_push.evaluate --policy random --seed 41000 --episodes 24 --output repro/baselines/random_dev
foreach ($policy in @('random', 'rule')) {
    & $py -m panda_push.evaluate --policy $policy --seed 61000 --episodes 100 --output "repro/baselines/${policy}_test"
    if ($LASTEXITCODE -ne 0) { throw "Baseline $policy failed" }
}
& $py -m panda_push.train --config configs/original.json --group sac_dense --seed 11 --steps 10000 --output repro/sac_dense/seed11_10k
& $py -m panda_push.train --resume repro/sac_dense/seed11_10k/final.zip --steps 40000 --output repro/sac_dense/seed11_50k
& $py -m panda_push.evaluate --checkpoint repro/sac_dense/seed11_50k/final.zip --seed 61000 --episodes 100 --output repro/test_50k/sac_dense_seed11
& $py -m panda_push.visualize cases --evaluation repro/test_50k/sac_dense_seed11 --output repro/test_50k/sac_dense_seed11/media
```

三组消融实验采用训练种子 11、29、47，统一 10,000 步。开发场景 41000–41023 用于跟踪学习曲线，独立测试场景 61000–61099 用于汇总。相同种子对应相同初始方块与目标；训练 seed 是组内统计重复单位。

```powershell
$groups = @('sac_dense', 'sac_sparse', 'sac_her')
$runs = @()
foreach ($seed in @(11, 29, 47)) {
    foreach ($group in $groups) {
        $run = "repro/$group/seed${seed}_10k"
        if (-not (Test-Path -LiteralPath "$run/training_summary.json")) {
            & $py -m panda_push.train --config configs/original.json --group $group --seed $seed --steps 10000 --output $run
            if ($LASTEXITCODE -ne 0) { throw "Training failed for $group seed $seed" }
        }
        & $py -m panda_push.evaluate --checkpoint "$run/final.zip" --seed 61000 --episodes 100 --output "repro/test_10k/${group}_seed${seed}"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $group seed $seed" }
        $recheck = "repro/safety_ablation/original_holdout_10k/${group}_seed${seed}"
        & $py -m panda_push.evaluate --checkpoint "$run/final.zip" --seed 81000 --episodes 100 --output $recheck
        if ($LASTEXITCODE -ne 0) { throw "Independent recheck failed for $group seed $seed" }
        $runs += $run
    }
}
& $py -m panda_push.study --root repro --output repro/study_10k
& $py -m panda_push.study --root repro --test-root repro/safety_ablation/original_holdout_10k --output repro/safety_ablation/original_study
& $py -m panda_push.visualize training --runs @runs --output repro/figures
```

物理泛化评估可用 `--physics idr`（中等扰动）、`ood`（质量、摩擦和边长同时增大）、`heavy`（只增质量）或 `slick`（摩擦降低）。`--physics idr` 也可用于训练。模型默认使用训练配置，评估可显式覆盖布局和物理属性。

```powershell
$evaluations = @('repro/test_50k/sac_dense_seed11')
foreach ($profile in @('idr', 'ood', 'heavy', 'slick')) {
    $out = "repro/generalization/seed11_$profile"
    & $py -m panda_push.evaluate --checkpoint repro/sac_dense/seed11_50k/final.zip --physics $profile --seed 61000 --episodes 100 --output $out
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $profile" }
    $evaluations += $out
}
& $py -m panda_push.visualize robustness --evaluations @evaluations --output repro/generalization
```

Panda 网格与原始 MJCF 附带 Menagerie 原始 Apache-2.0 许可，详见 [模型许可](models/franka_emika_panda/LICENSE) 和 [来源记录](docs/code-source.json)。


## 复现安全终止代价对照

原始稀疏奖励在目标外每步扣 1 分，提前发生安全终止的回合会少扣分。修正版本以 `--sparse-safety-cost` 对这一事件计入剩余步数的代价。B、C 使用同一个修正，HER 重标时也保留这项代价。修正后的决策在新的测试种子 81000–81099 上验证。上一节生成同一批种子上的原始模型重评，用于比较前后的真实成功率和提前终止比例。

```powershell
$safetyRuns = @()
foreach ($seed in @(11, 29, 47)) {
    foreach ($group in @('sac_sparse', 'sac_her')) {
        $out = "repro/safety_ablation/$group/seed${seed}_10k"
        & $py -m panda_push.train --group $group --sparse-safety-cost --seed $seed --steps 10000 --output $out
        if ($LASTEXITCODE -ne 0) { throw "Revised training failed for $group seed $seed" }
        & $py -m panda_push.evaluate --checkpoint "$out/final.zip" --seed 81000 --episodes 100 --output "repro/safety_ablation/revised_holdout_10k/${group}_seed${seed}"
        if ($LASTEXITCODE -ne 0) { throw "Revised evaluation failed for $group seed $seed" }
        $safetyRuns += "repro/$group/seed${seed}_10k"
        $safetyRuns += $out
    }
}
& $py -m panda_push.study --root repro/safety_ablation --test-root repro/safety_ablation/revised_holdout_10k --groups sac_sparse sac_her --output repro/safety_ablation/revised_study
& $py -m panda_push.visualize training --runs @safetyRuns --output repro/safety_ablation/figures
```


## 复现物理随机化训练

下面训练具有相同优化预算的中等物理随机化模型，然后在三种物理条件下与基准模型匹配场景比较。

```powershell
$out = 'repro/generalization/idr_training'
& $py -m panda_push.train --config configs/original.json --group sac_dense --physics idr --seed 11 --steps 10000 --output "$out/seed11_10k"
$evals = @()
foreach ($profile in @('nominal','idr','ood')) {
    foreach ($source in @('nominal','idr')) {
        $checkpoint = if ($source -eq 'nominal') {
            'repro/sac_dense/seed11_10k/final.zip'
        } else {
            "$out/seed11_10k/final.zip"
        }
        $evaluation = "$out/${source}_trained_$profile"
        & $py -m panda_push.evaluate --checkpoint $checkpoint --physics $profile --seed 61000 --episodes 100 --output $evaluation
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $source / $profile" }
        $evals += $evaluation
    }
}
& $py -m panda_push.visualize robustness --evaluations @evals --output "$out/figures"
```


可在相同组合物理条件下评估另外两个标称训练种子。seed 11 的对照已使用该条件，下面补充 29 与 47。

```powershell
foreach ($seed in @(29, 47)) {
    & $py -m panda_push.evaluate --checkpoint "repro/sac_dense/seed${seed}_10k/final.zip" --physics ood --seed 61000 --episodes 100 --output "repro/generalization/seed${seed}_10k_ood"
    if ($LASTEXITCODE -ne 0) { throw "OOD evaluation failed for seed $seed" }
}
```

## 复现接触安全续实验

冻结原 50,000 步模型并重放单因素诊断。然后分别运行 Worker 0、1、2，每个负责三个匹配的训练组合。可以在三个 PowerShell 终端并行运行。训练结果沿用相同的参数和数据路径，最后生成配对汇总及图表。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m panda_push.contact_study --checkpoint results/sac_dense/seed11_50k/final.zip --seed 92000 --episodes 100 --output repro/contact_safety/diagnostics
.\scripts\run_contact_safety.ps1 -Worker 0 -OutputRoot repro/contact_safety
.\scripts\run_contact_safety.ps1 -Worker 1 -OutputRoot repro/contact_safety
.\scripts\run_contact_safety.ps1 -Worker 2 -OutputRoot repro/contact_safety
& $py -m panda_push.contact_results --root repro/contact_safety
```

## 接触限速策略续训

从上一轮三个限速模型各自的 50,000 步检查点续训至 100,000 步。下面使用新目录复现训练和六个检查点的配对留出评估。三个训练命令可以在三个 PowerShell 进程中并行运行。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/success_progress'
foreach ($seed in 11, 29, 47) {
    .\scripts\run_success_progress.ps1 -Seed $seed -OutputRoot $out
}
$manifest = Get-Content 'results/success_progress/manifest.json' -Raw | ConvertFrom-Json
foreach ($entry in $manifest.models) {
    if ($entry.arm -eq 'speed_100k') {
        $run = "$out/training/seed$($entry.seed)"
        $entry.checkpoint = "$run/final.zip"
        $entry.config = "$run/config.json"
    }
}
$manifestPath = "$out/manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content $manifestPath -Encoding utf8
foreach ($entry in $manifest.models) {
    & $py -m panda_push.success_eval run --manifest $manifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $manifestPath
$runs = @(11, 29, 47 | ForEach-Object { "$out/training/seed$_" })
& $py -m panda_push.visualize training --runs @runs --output "$out/learning"
```

用选出的 100,000 步模型重放组合物理场景 114003。

```powershell
.\.venv\Scripts\python.exe -m panda_push.evaluate --checkpoint results/success_progress/training/seed29/final.zip --physics ood --seed 114003 --episodes 1 --viewer
```

## 种子 47 失败复盘与安全约束筛选

从项目根目录运行。复盘读取上一轮的回合记录，并检验 99 次失败的物理重放和末误差。三个续训作业可分别在三个终端运行。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/success_90'
& $py -m panda_push.failure_diagnostics --checkpoint results/success_progress/training/seed47/final.zip --episodes results/success_progress/test/speed_100k/seed47/episodes.csv --output "$out/seed47_diagnostics"
foreach ($seed in 11, 47) {
    & $py -m panda_push.train --resume "results/success_progress/training/seed$seed/final.zip" --continuation-seed (142000 + $seed) --steps 50000 --output "$out/training/seed$seed"
    if ($LASTEXITCODE -ne 0) { throw "Continuation failed for $seed" }
}
& $py -m panda_push.train --resume results/success_progress/training/seed47/final.zip --train-layout focused --continuation-seed 142147 --steps 30000 --output "$out/training/focused_seed47"
if ($LASTEXITCODE -ne 0) { throw 'Focused continuation failed' }

foreach ($phase in 'archived', 'intermediate', 'continued', 'focused') {
    foreach ($seed in 11, 29, 47) {
        & $py -m panda_push.success_target --root $out screen --phase $phase --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Development evaluation failed for $phase seed $seed" }
    }
}
& $py -m panda_push.success_target --root $out select
if ($LASTEXITCODE -ne 0) { throw 'Candidate selection failed' }
$manifestPath = "$out/manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.models) {
    & $py -m panda_push.success_eval run --manifest $manifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $manifestPath
```

在同一复现目录里运行第二批独立场景，额外比较单回合最大力。

```powershell
$peak = Get-Content 'results/success_90/peak_confirm/manifest.json' -Raw | ConvertFrom-Json
foreach ($entry in $peak.models) {
    $entry.checkpoint = $entry.checkpoint -replace '^results/success_90/', "$out/"
    $entry.config = $entry.config -replace '^results/success_90/', "$out/"
}
New-Item -ItemType Directory -Path "$out/peak_confirm" -Force | Out-Null
$peakPath = "$out/peak_confirm/manifest.json"
$peak | ConvertTo-Json -Depth 6 | Set-Content $peakPath -Encoding utf8
foreach ($entry in $peak.models) {
    & $py -m panda_push.success_eval run --manifest $peakPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Peak confirmation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $peakPath
```

## 超过 95% 的复验

从种子 47 的 150,000 步模型追加 40,000 步，并提高组合物理变化的训练权重。同一套命令复现开发集选型及全新留出评估。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/success_95'
$study = 'configs/success_95.json'
& $py -m panda_push.train --resume results/success_90/training/seed47/final.zip --train-physics contact_mix_ood --continuation-seed 164047 --steps 40000 --output "$out/training/seed47_ood"
if ($LASTEXITCODE -ne 0) { throw 'OOD continuation failed' }
foreach ($seed in 11, 29, 47) {
    & $py -m panda_push.success_target --root $out --study $study screen --phase archived --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Archived model screening failed for seed $seed" }
}
foreach ($phase in 'intermediate', 'continued') {
    & $py -m panda_push.success_target --root $out --study $study screen --phase $phase --seed 47
    if ($LASTEXITCODE -ne 0) { throw "Continued model screening failed at $phase" }
}
& $py -m panda_push.success_target --root $out --study $study select
if ($LASTEXITCODE -ne 0) { throw 'Candidate selection failed' }
$manifestPath = "$out/manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.models) {
    & $py -m panda_push.success_eval run --manifest $manifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $manifestPath
```

## 进圈后保持与组合变化复验

复现种子 47 离圈诊断，再在组合变化开发场景筛选推理控制。该实验使用原检查点。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/success_hold'
$study = 'configs/success_hold.json'
& $py -m panda_push.failure_diagnostics --checkpoint results/success_95/training/seed47_ood/checkpoint_170000.zip --config results/success_95/development/configs/170k_ood_42_26/seed47.json --episodes results/success_95/test/peak_matched/seed47/episodes.csv --output "$out/prior_seed47"
if ($LASTEXITCODE -ne 0) { throw 'Failure replay diverged' }
foreach ($seed in 11, 29, 47) {
    & $py -m panda_push.success_target --root $out --study $study screen --phase archived --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Development screening failed for $seed" }
}
& $py -m panda_push.success_target --root $out --study $study select
if ($LASTEXITCODE -ne 0) { throw 'Candidate selection failed' }
$manifestPath = "$out/manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.models) {
    & $py -m panda_push.success_eval run --manifest $manifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $manifestPath
```

## 进圈后低速微调

从项目根目录运行。微调试验沿用当前三个 SAC 检查点，记录 3 cm 主成功率以及 1 cm、2 cm 到位率。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/success_precision'
$study = 'configs/success_precision.json'
foreach ($seed in 11, 29, 47) {
    & $py -m panda_push.success_target --root $out --study $study screen --phase archived --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Development evaluation failed for seed $seed" }
}
foreach ($seed in 29, 47) {
    & $py -m panda_push.success_target --root $out --study $study screen --phase focused --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Projected centering evaluation failed for seed $seed" }
}
foreach ($seed in 11, 29, 47) {
    & $py -m panda_push.success_target --root $out --study $study screen --phase intermediate --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Speed limit evaluation failed for seed $seed" }
}
& $py -m panda_push.success_target --root $out --study $study select
if ($LASTEXITCODE -ne 0) { throw 'Centering selection failed' }
$manifestPath = "$out/manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.models) {
    & $py -m panda_push.success_eval run --manifest $manifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $manifestPath
```

追加的力余量确认使用另一批独立场景，可沿用同一复现目录中的开发配置。

```powershell
$confirmation = Get-Content 'results/success_precision/force_confirmation/manifest.json' -Raw | ConvertFrom-Json
foreach ($entry in $confirmation.models) {
    $entry.config = $entry.config -replace '^results/success_precision/', "$out/"
}
New-Item -ItemType Directory -Path "$out/force_confirmation" -Force | Out-Null
$confirmationPath = "$out/force_confirmation/manifest.json"
$confirmation | ConvertTo-Json -Depth 6 | Set-Content $confirmationPath -Encoding utf8
foreach ($entry in $confirmation.models) {
    & $py -m panda_push.success_eval run --manifest $confirmationPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Force confirmation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $confirmationPath
```

## 种子 29 的进圈前横向推偏

从项目根目录运行。两条续训都使用原 50,000 步模型和缓冲区，场景权重只用于训练。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/success_seed29'
$study = 'configs/success_seed29.json'
$previous = 'results/success_precision/force_confirmation/test/guarded_centering/seed29/episodes.csv'
New-Item -ItemType Directory -Path $out -Force | Out-Null
Import-Csv $previous | Where-Object profile -eq 'ood' |
    Export-Csv -LiteralPath "$out/reference_ood.csv" -NoTypeInformation -Encoding utf8
& $py -m panda_push.failure_diagnostics --checkpoint results/contact_safety/training/speed_limit/seed29/final.zip --config results/success_precision/development/configs/50k_project12/seed29.json --episodes "$out/reference_ood.csv" --output "$out/diagnosis"
if ($LASTEXITCODE -ne 0) { throw 'Failure replay diverged' }
& $py -m panda_push.train --resume results/contact_safety/training/speed_limit/seed29/final.zip --train-physics contact_mix_ood --continuation-seed 173029 --steps 30000 --output "$out/training/seed29_ood"
if ($LASTEXITCODE -ne 0) { throw 'OOD continuation failed' }
& $py -m panda_push.train --resume results/contact_safety/training/speed_limit/seed29/final.zip --train-physics contact_mix_ood --train-layout focused --continuation-seed 173129 --steps 20000 --output "$out/training/seed29_layout_ood"
if ($LASTEXITCODE -ne 0) { throw 'Focused layout continuation failed' }
foreach ($phase in 'archived', 'approach', 'intermediate', 'focused', 'continued', 'layout') {
    & $py -m panda_push.success_target --root $out --study $study screen --phase $phase --seed 29
    if ($LASTEXITCODE -ne 0) { throw "Development screening failed at $phase" }
}
$extended = Get-Content 'results/success_seed29/development_confirmation/manifest.json' -Raw | ConvertFrom-Json
foreach ($entry in $extended.models) {
    $entry.checkpoint = $entry.checkpoint -replace '^results/success_seed29/', "$out/"
    $entry.config = $entry.config -replace '^results/success_seed29/', "$out/"
}
New-Item -ItemType Directory -Path "$out/development_confirmation" -Force | Out-Null
$extendedPath = "$out/development_confirmation/manifest.json"
$extended | ConvertTo-Json -Depth 6 | Set-Content $extendedPath -Encoding utf8
foreach ($entry in $extended.models) {
    & $py -m panda_push.success_eval run --manifest $extendedPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Extended development evaluation failed for $($entry.arm)" }
}
& $py -m panda_push.success_target --root $out --study $study select
if ($LASTEXITCODE -ne 0) { throw 'Seed 29 selection failed' }
$manifestPath = "$out/manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.models) {
    & $py -m panda_push.success_eval run --manifest $manifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $manifestPath
```

第二批复验从已归档的 70,000 步模型加入负向目标加权。

```powershell
$next = "$out/followup"
$nextStudy = 'configs/success_seed29_negative.json'
& $py -m panda_push.train --resume results/success_seed29/training/seed29_layout_ood/final.zip --train-physics contact_mix_ood --train-layout negative_y --continuation-seed 173229 --steps 20000 --output "$next/training/seed29_negative"
if ($LASTEXITCODE -ne 0) { throw 'Negative-direction continuation failed' }
foreach ($phase in 'archived', 'approach', 'intermediate', 'continued') {
    & $py -m panda_push.success_target --root $next --study $nextStudy screen --phase $phase --seed 29
    if ($LASTEXITCODE -ne 0) { throw "Follow-up development screening failed at $phase" }
}
& $py -m panda_push.success_target --root $next --study $nextStudy select
if ($LASTEXITCODE -ne 0) { throw 'Follow-up model selection failed' }
$nextManifestPath = "$next/manifest.json"
$nextManifest = Get-Content $nextManifestPath -Raw | ConvertFrom-Json
foreach ($entry in $nextManifest.models) {
    & $py -m panda_push.success_eval run --manifest $nextManifestPath --arm $entry.arm --seed $entry.seed
    if ($LASTEXITCODE -ne 0) { throw "Follow-up held-out evaluation failed for $($entry.arm) seed $($entry.seed)" }
}
& $py -m panda_push.success_eval aggregate --manifest $nextManifestPath
```

## 首次接触侧偏的小规模消融

对照和三个干预组均加载已归档的种子 29 的 90,000 步候选。实验只运行评估，`probe` 重放 13 个已知失败及 13 个匹配成功，`heldout` 使用新的 96 个组合物理变化场景。每个组输出协议、回合指标与控制步轨迹。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/lateral_ablation'
foreach ($dataset in 'probe', 'heldout') {
    foreach ($arm in 'baseline', 'before_contact', 'early_contact', 'both') {
        & $py -m panda_push.lateral_ablation --output $out --dataset $dataset --arm $arm
        if ($LASTEXITCODE -ne 0) { throw "Lateral ablation failed for $dataset $arm" }
    }
}
```

## 大横向偏置时的首次接触前门控

沿用同一冻结检查点，将仅在首次接触前大偏置且接近方块时激活的门控修正，与无修正基线在两批各 128 个组合场景逐回合比较。阈值及场景种子见[冻结协议](results/success_seed29/gated_lateral_ablation/plan.json)。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/gated_lateral_ablation'
foreach ($dataset in 'gated_a', 'gated_b') {
    foreach ($arm in 'baseline', 'gated_before_contact') {
        & $py -m panda_push.lateral_ablation --output $out --dataset $dataset --arm $arm
        if ($LASTEXITCODE -ne 0) { throw "Gated lateral ablation failed for $dataset $arm" }
    }
}
```

## 主动对齐方块后方再交给 SAC

同一个 90,000 步检查点下，对比原 SAC 与首次接触前的几何对齐后交接。先重放旧诊断回合，再做两批独立组合物理场景配对。所有配置、代码指纹及场景种子见[冻结配置](results/success_seed29/aligned_handoff_ablation/plan.json)。

```powershell
$py = '.\.venv\Scripts\python.exe'
$out = 'repro/aligned_handoff_ablation'
foreach ($dataset in 'probe', 'align_a', 'align_b') {
    foreach ($arm in 'baseline', 'aligned_handoff') {
        & $py -m panda_push.lateral_ablation --output $out --dataset $dataset --arm $arm
        if ($LASTEXITCODE -ne 0) { throw "Aligned handoff failed for $dataset $arm" }
    }
}
```
