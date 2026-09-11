# S2V Object-only 多视角流程生成

流程：从 easy-HOI 的 `index.jsonl` 直接生成完整 multi-view jobs，然后用 HunyuanImage 3 Distil 推理。

## 入口

| 阶段 | 入口 | 作用 |
|---|---|---|
| 生成 jobs | `build_s2v_multiview_jobs.py` | 从gemma标注的结果中，每条视频生成 anchor + extra views |
| 推理 | `run_s2v_multiview_hunyuan_distil.py` | 单 worker 推理、dry-run 和分片执行 |
| 多卡启动 | `run_s2v_multiview_8gpu.sh` | 启动多个独立 GPU worker |

## 环境和数据要求

推荐环境与主仓库一致：Python 3.12、CUDA 12.8、PyTorch 2.8。安装项目依赖：

```bash
pip install -r requirements.txt
```

运行前需要准备：

1. easy-HOI `index.jsonl`。每行至少应包含 `sample_id`、`state`、`candidate_frames`、`frame_detections`、`references` 和 `record_json`。
2. index 和 `record_json` 中引用的候选帧必须在当前机器可访问。
3. 本地 distilled checkpoint，默认路径为：

```text
../weights/HunyuanImage-3.0-Instruct-Distil
```

checkpoint 必须包含 32 个 safetensors shard，以及 `config.json`、`generation_config.json`、`model.safetensors.index.json` 和 `tokenizer.json`。

数据、模型、生成图片、manifest 和日志都不提交 Git。仓库的 `.gitignore` 已排除 `dataset/`、`weights/` 和输出目录。

## 第一步：生成完整 jobs

下面这一条命令完成全部 jobs 构建：

```bash
.venv/bin/python build_s2v_multiview_jobs.py \
  --index /path/to/easy_hoi/index.jsonl \
  --output dataset/s2v_keyframe_recontext/jobs.multiview.jsonl \
  --views-per-video 4 \
  --workers 8
```

默认每个有效视频输出四行：一个 anchor 和三个 complementary views。每行都可独立推理，同时带有以下关联字段：

- `sample_id`：视频分组键。
- `entity_id`：同一视频中持续跟踪的目标物体。
- `view_rank=0`、`is_anchor=true`：anchor view。
- `view_rank=1..N-1`、`is_anchor=false`：额外视角。
- `parent_job_name`、`anchor_frame_index`：回溯 anchor。
- `bbox_source`：bbox 来自帧检测、上游 view 或插值。
- `quality_warning`、`view_score`：候选质量与视角差异审计信息。

构建规则：

- 先从 index 做 anchor 质量过滤，再按类别 round-robin，避免热门类别淹没结果。
- extra view 必须属于同一个 `sample_id + entity_id`。
- 视角选择同时考虑 bbox 面积、清晰度、置信度和图像差异。
- 一个视频要么写出恰好 `--views-per-video` 行，要么整组不写，避免残缺组。
- 输出先写入 `.tmp`，全部完成后再原子替换目标文件。

正式构建前先跑小样本：

```bash
.venv/bin/python build_s2v_multiview_jobs.py \
  --index /path/to/easy_hoi/index.jsonl \
  --output /tmp/jobs.multiview.smoke.jsonl \
  --scan-limit 1000 \
  --target 20 \
  --views-per-video 4 \
  --workers 4
```

重点检查终端中的：

- `selected anchors`：anchor 筛选数量。
- `Video statuses`：成功、视角不足、差异不足的视频数。
- `Candidate rejections`：bbox、清晰度、遮边等统计。

常用调节参数：

- `--target N`：类别均衡后最多选择 N 个视频；`0` 表示全部。
- `--per-category N`：每个物体类别最多 N 个视频；`0` 不限制。
- `--views-per-video N`：每视频总 view 数，包含 anchor。
- `--min-diversity X`：最小视角差异，默认 `0.05`；提高会减少近重复 view，但也会降低视频通过率。
- `--strict-quality`：把 extra-view 质量警告改成硬拒绝。
- `--no-estimated-bbox`：检测缺失时不允许用上游或插值 bbox。
- `--limit N`：仅处理筛选后的前 N 个 anchor，适合调试。

`--anchors existing.jsonl` 是故障恢复入口，用于复用已经筛好的 anchor；它不是正常新流程的必需步骤，并且不能与 `--index` 同时传入。

## 第二步：推理前 dry-run

dry-run 会验证 checkpoint、jobs、图片路径、bbox crop、输出命名和 manifest，但不会加载模型执行生成：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
.venv/bin/python run_s2v_multiview_hunyuan_distil.py \
  --jobs dataset/s2v_keyframe_recontext/jobs.multiview.jsonl \
  --output-root /tmp/s2v_multiview_dryrun \
  --limit 8 \
  --dry-run
```

确认 `manifest.dryrun.shard_00.jsonl` 中：

- `view_rank` 完整且输出路径不重复。
- `input_path` 和 `model_input_path` 正确。
- `crop_box`、`model_input_size` 与原图尺寸合理。
- `steps` 为 8。

Distil MeanFlow checkpoint 固定使用 8 steps，脚本会拒绝其他 `--diff-infer-steps`。

## 第三步：8 GPU 正式运行

默认配置是四个 worker，每个 worker 使用两张卡：

```bash
JOBS="$PWD/dataset/s2v_keyframe_recontext/jobs.multiview.jsonl" \
OUTPUT_BASE="$PWD/dataset/s2v_object_only-hunyuan-distil/multiview" \
bash run_s2v_multiview_8gpu.sh
```

默认 GPU 分组为：

```text
0,1  2,3  4,5  6,7
```

可通过环境变量调整：

```bash
GPU_PAIRS="0,1 2,3" \
JOBS=/path/to/jobs.multiview.jsonl \
OUTPUT_BASE=/path/to/output \
WAIT_FOR_WORKERS=1 \
bash run_s2v_multiview_8gpu.sh
```

环境变量说明：

- `PYTHON`：Python 路径，默认 `$ROOT/.venv/bin/python`。
- `MODEL_ID`：本地 distilled checkpoint 路径。
- `JOBS`：完整 multi-view jobs JSONL。
- `OUTPUT_BASE`：推理输出根目录。
- `GPU_PAIRS`：以空格分隔的 worker GPU 集合；worker 数就是 shard 数。
- `RESERVE_GIB`：每张卡为激活和其他进程预留的显存，默认 14 GiB。
- `STAGGER_SECONDS`：worker 启动间隔，默认 20 秒。
- `WAIT_FOR_WORKERS=1`：前台等待全部 worker，建议调度系统中启用。
- `EXTRA`：追加推理参数，例如 `EXTRA="--save-native --no-compare"`。

同一视频的所有 views 会被分到同一个 shard。启动器将日志和 PID 写入 `OUTPUT_BASE/logs/`。

## 输出与断点续跑

```text
OUTPUT_BASE/
  logs/
  videos/<sample_id>/
    edited/
    reference_inputs/
    compare/
    native/                 # 仅 --save-native
  manifest.shard_XX.jsonl
  invalid_input.shard_XX.jsonl
  review_required.shard_XX.jsonl
```

正常重启相同命令即可断点续跑：manifest 中已成功的 job 和已存在的输出会被跳过。只有明确需要全部重做时才使用 `EXTRA="--overwrite"`。
