import csv
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from flask import Blueprint, jsonify, request, send_from_directory
from ultralytics import YOLO

from utils_paths import resolve_existing_file


CTX = None
bp = Blueprint("train", __name__)


def init(ctx):
    global CTX
    CTX = ctx
    _patch_ultralytics_checkpoint_io()


@bp.before_request
def _require_auth():
    if CTX is None:
        return None
    return CTX.enforce_request_auth()


training_status_lock = threading.Lock()
validation_status_lock = threading.Lock()
conversion_status_lock = threading.Lock()

training_status_by_user = {}
validation_status_by_user = {}
conversion_status_by_user = {}
STATUS_RETENTION_SECONDS = 1800
ACTIVE_STATUS_RETENTION_SECONDS = 86400
_ultralytics_checkpoint_patch_lock = threading.Lock()
_ultralytics_checkpoint_patch_done = False
_AUTO_DATASET_NAME = "annotation_live"


def _ensure_path_under_current_dir(target_path, field_name="path"):
    target = Path(str(target_path)).expanduser()
    if not target.is_absolute():
        target = (Path(CTX.CURRENT_DIR) / target).resolve()
    else:
        target = target.resolve()
    root = Path(CTX.CURRENT_DIR).resolve()
    try:
        target.relative_to(root)
    except Exception as exc:
        raise ValueError(f"{field_name} 必须位于当前工作区内") from exc
    return target


def _empty_training_status():
    return {"status": "idle", "progress": 0, "message": "", "owner": ""}


def _empty_validation_status():
    return {"status": "idle", "progress": 0, "message": "", "owner": ""}


def _empty_conversion_status():
    return {"status": "idle", "progress": 0, "message": "", "owner": ""}


def _status_owner_key(username):
    return str(username or "").strip() or "__anonymous__"


def _touch_status_record(record):
    if record is not None:
        record["_updated_at"] = float(time.time())
    return record


def _status_ttl_seconds(record):
    status = str((record or {}).get("status") or "").strip().lower()
    if status in {"training", "validating", "converting", "starting"}:
        return ACTIVE_STATUS_RETENTION_SECONDS
    return STATUS_RETENTION_SECONDS


def _cleanup_status_store(store):
    now = float(time.time())
    stale_keys = []
    for key, record in list(store.items()):
        updated_at = float((record or {}).get("_updated_at") or 0.0)
        ttl = _status_ttl_seconds(record)
        if updated_at <= 0:
            if ttl <= STATUS_RETENTION_SECONDS:
                stale_keys.append(key)
            continue
        if now - updated_at > ttl:
            stale_keys.append(key)
    for key in stale_keys:
        store.pop(key, None)


def _public_status_payload(record, fallback_factory):
    data = dict(record or fallback_factory())
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def _paths_overlap(path_a, path_b):
    try:
        left = Path(str(path_a)).resolve()
        right = Path(str(path_b)).resolve()
        return left == right or left.is_relative_to(right) or right.is_relative_to(left)
    except Exception:
        try:
            left = str(Path(str(path_a)).resolve()).lower()
            right = str(Path(str(path_b)).resolve()).lower()
            return left == right or left.startswith(right + os.sep.lower()) or right.startswith(left + os.sep.lower())
        except Exception:
            return False


def _predict_conversion_output_path(status_record):
    fmt = _normalize_export_format((status_record or {}).get("format"))
    model_path = str((status_record or {}).get("model_path") or "").strip()
    output_path_requested = (status_record or {}).get("output_path_requested")
    if not fmt or not model_path:
        return ""
    try:
        return str(_resolve_conversion_output_path(output_path_requested, model_path, fmt))
    except Exception:
        return ""


def _get_status_record(store, username, factory, create=False):
    _cleanup_status_store(store)
    key = _status_owner_key(username)
    record = store.get(key)
    if record is None and create:
        record = factory()
        record["owner"] = str(username or "")
        store[key] = record
    return _touch_status_record(record)


def _get_training_status_ref(username, create=False):
    return _get_status_record(training_status_by_user, username, _empty_training_status, create=create)


def _get_validation_status_ref(username, create=False):
    return _get_status_record(validation_status_by_user, username, _empty_validation_status, create=create)


def _get_conversion_status_ref(username, create=False):
    return _get_status_record(conversion_status_by_user, username, _empty_conversion_status, create=create)


def _current_username():
    try:
        return str(CTX.get_current_username() or "").strip()
    except Exception:
        return ""


def _current_workspace_dir():
    try:
        return str(CTX.CURRENT_DIR or "").strip()
    except Exception:
        return ""


def _normalize_task_type(value):
    raw = str(value or "").strip().lower()
    if raw in ("segment", "seg", "mask", "polygon"):
        return "segment"
    if raw in ("detect", "detection", "box", "bbox", "recognition", "recognize"):
        return "detect"
    return ""


def _current_task_type():
    try:
        return _normalize_task_type(CTX.get_current_task() if CTX is not None else "")
    except Exception:
        return ""


def _preferred_metric_scope(task_type):
    return "mask" if _normalize_task_type(task_type) == "segment" else "box"


def _train_loss_candidates(task_type):
    if _normalize_task_type(task_type) == "segment":
        return ["seg", "box", "cls", "dfl"]
    return ["box", "seg", "cls", "dfl"]


def _csv_metric_candidates(task_type, metric_kind):
    is_segment = _normalize_task_type(task_type) == "segment"
    scope_primary = "M" if is_segment else "B"
    scope_fallback = "B" if is_segment else "M"
    if metric_kind == "train_loss":
        primary = "seg_loss" if is_segment else "box_loss"
        fallback = "box_loss" if is_segment else "seg_loss"
        return [f"train/{primary}", primary, f"train/{fallback}", fallback, "loss"]
    if metric_kind == "map50":
        return [f"metrics/mAP50({scope_primary})", f"metrics/mAP50({scope_fallback})", "metrics/mAP50", "mAP50"]
    if metric_kind == "map":
        return [
            f"metrics/mAP50-95({scope_primary})",
            f"metrics/mAP({scope_primary})",
            f"metrics/mAP50-95({scope_fallback})",
            f"metrics/mAP({scope_fallback})",
            "metrics/mAP50-95",
            "mAP",
        ]
    if metric_kind == "precision":
        return [f"metrics/precision({scope_primary})", f"metrics/precision({scope_fallback})", "metrics/precision", "precision"]
    if metric_kind == "recall":
        return [f"metrics/recall({scope_primary})", f"metrics/recall({scope_fallback})", "metrics/recall", "recall"]
    return []


def _extract_training_loss_value(trainer, task_type=""):
    train_loss = getattr(trainer, "train_loss", None)
    if train_loss is None:
        return None
    for attr in _train_loss_candidates(task_type):
        if hasattr(train_loss, attr):
            try:
                value = float(getattr(train_loss, attr))
                if value == value:
                    return value
            except Exception:
                continue
    try:
        value = float(train_loss)
        if value == value:
            return value
    except Exception:
        pass
    return None


def _resolve_results_metric_container(results, task_type=""):
    is_segment = _normalize_task_type(task_type) == "segment"
    order = [("seg", "mask"), ("box", "box")] if is_segment else [("box", "box"), ("seg", "mask")]
    for attr, scope in order:
        container = getattr(results, attr, None)
        if container is None:
            continue
        if any(hasattr(container, key) for key in ("map", "map50", "mp", "mr")):
            return container, scope
    for attr, scope in order[::-1]:
        container = getattr(results, attr, None)
        if container is not None:
            return container, scope
    return None, _preferred_metric_scope(task_type)


def _record_owned_by_current_user(record):
    username = _current_username()
    try:
        owner = str((record or {}).get("owner") or "").strip()
    except Exception:
        owner = ""
    if not owner:
        return True
    return bool(username) and owner == username


def _safe_replace_file_bytes(target_path, data, retries=6, retry_delay=0.15):
    target = Path(str(target_path)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(max(1, int(retries))):
        tmp_path = (target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp").resolve()
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(str(tmp_path), str(target))
            return target
        except Exception as e:
            last_err = e
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if attempt + 1 < max(1, int(retries)):
                time.sleep(float(retry_delay) * (attempt + 1))
    if last_err is not None:
        raise last_err
    return target


def _safe_torch_save_file(obj, target_path, retries=6, retry_delay=0.15):
    target = Path(str(target_path)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(max(1, int(retries))):
        tmp_path = (target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp").resolve()
        try:
            torch.save(obj, str(tmp_path))
            os.replace(str(tmp_path), str(target))
            return target
        except Exception as e:
            last_err = e
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if attempt + 1 < max(1, int(retries)):
                time.sleep(float(retry_delay) * (attempt + 1))
    if last_err is not None:
        raise last_err
    return target


def _patch_ultralytics_checkpoint_io():
    global _ultralytics_checkpoint_patch_done
    if _ultralytics_checkpoint_patch_done:
        return
    with _ultralytics_checkpoint_patch_lock:
        if _ultralytics_checkpoint_patch_done:
            return
        try:
            import ultralytics.engine.trainer as trainer_mod  # noqa: PLC0415
            import ultralytics.utils.torch_utils as torch_utils_mod  # noqa: PLC0415
        except Exception:
            return

        if getattr(trainer_mod.BaseTrainer.save_model, "__name__", "") != "_patched_save_model":
            def _patched_save_model(self):
                import io

                buffer = io.BytesIO()
                trainer_mod.torch.save(
                    {
                        "epoch": self.epoch,
                        "best_fitness": self.best_fitness,
                        "model": None,
                        "ema": deepcopy(trainer_mod.unwrap_model(self.ema.ema)).half(),
                        "updates": self.ema.updates,
                        "optimizer": trainer_mod.convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                        "scaler": self.scaler.state_dict(),
                        "train_args": vars(self.args),
                        "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                        "train_results": self.read_results_csv(),
                        "date": datetime.datetime.now().isoformat(),
                        "version": trainer_mod.__version__,
                        "git": {
                            "root": str(trainer_mod.GIT.root),
                            "branch": trainer_mod.GIT.branch,
                            "commit": trainer_mod.GIT.commit,
                            "origin": trainer_mod.GIT.origin,
                        },
                        "license": "AGPL-3.0 (https://ultralytics.com/license)",
                        "docs": "https://docs.ultralytics.com",
                    },
                    buffer,
                )
                serialized_ckpt = buffer.getvalue()
                self.wdir.mkdir(parents=True, exist_ok=True)
                _safe_replace_file_bytes(self.last, serialized_ckpt)
                if self.best_fitness == self.fitness:
                    _safe_replace_file_bytes(self.best, serialized_ckpt)
                if (self.save_period > 0) and (self.epoch % self.save_period == 0):
                    _safe_replace_file_bytes(self.wdir / f"epoch{self.epoch}.pt", serialized_ckpt)

            trainer_mod.BaseTrainer.save_model = _patched_save_model

        if getattr(trainer_mod.strip_optimizer, "__name__", "") != "_patched_strip_optimizer":
            def _patched_strip_optimizer(f="best.pt", s="", updates=None):
                try:
                    ckpt = torch_utils_mod.torch_load(f, map_location=torch.device("cpu"))
                    assert isinstance(ckpt, dict), "checkpoint is not a Python dictionary"
                    assert "model" in ckpt, "'model' missing from checkpoint"
                except Exception as e:
                    torch_utils_mod.LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
                    return {}

                metadata = {
                    "date": datetime.datetime.now().isoformat(),
                    "version": torch_utils_mod.__version__,
                    "license": "AGPL-3.0 License (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                }

                if ckpt.get("ema"):
                    ckpt["model"] = ckpt["ema"]
                if hasattr(ckpt["model"], "args"):
                    ckpt["model"].args = dict(ckpt["model"].args)
                if hasattr(ckpt["model"], "criterion"):
                    ckpt["model"].criterion = None
                ckpt["model"].half()
                for p in ckpt["model"].parameters():
                    p.requires_grad = False

                args = {**torch_utils_mod.DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}
                for k in ("optimizer", "best_fitness", "ema", "updates", "scaler"):
                    ckpt[k] = None
                ckpt["epoch"] = -1
                ckpt["train_args"] = {k: v for k, v in args.items() if k in torch_utils_mod.DEFAULT_CFG_KEYS}

                combined = {**metadata, **ckpt, **(updates or {})}
                output_path = Path(str(s or f)).resolve()
                _safe_torch_save_file(combined, output_path)
                mb = os.path.getsize(str(output_path)) / 1e6
                torch_utils_mod.LOGGER.info(
                    f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB"
                )
                return combined

            torch_utils_mod.strip_optimizer = _patched_strip_optimizer
            trainer_mod.strip_optimizer = _patched_strip_optimizer

        _ultralytics_checkpoint_patch_done = True


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_model_type(text):
    s = str(text or "").strip().lower()
    if not s:
        return None
    name = s.replace("\\", "/").split("/")[-1]
    m = re.search(r"yolo(?:v)?(\d+)", name) or re.search(r"yolo(?:v)?(\d+)", s)
    if not m:
        return None
    return f"yolov{m.group(1)}"


def _normalize_train_split(train_split):
    try:
        value = int(train_split)
    except (TypeError, ValueError):
        value = 80
    return min(95, max(50, value))


def _normalize_export_format(export_format):
    value = str(export_format or "").strip().lower()
    if value in ["onnx"]:
        return "onnx"
    if value in ["tensorrt", "trt", "engine"]:
        return "engine"
    if value in ["rknn", "rockchip"]:
        return "rknn"
    if value in ["bmodel", "sophgo", "tpu-mlir", "tpu_mlir"]:
        return "bmodel"
    return None


def _normalize_precision(value):
    text = str(value or "fp32").strip().lower()
    if text in ["fp16", "float16", "half"]:
        return "fp16"
    if text in ["int8"]:
        return "int8"
    return "fp32"


def _rknn_onnx_compat_message(version_text):
    v = str(version_text or "").strip() or "unknown"
    return (
        f"RKNN 与当前 onnx=={v} 不兼容：rknn-toolkit2 2.3.2 仍依赖 onnx.mapping。"
        f" 请将 RKNN 转换环境中的 onnx 降级到 1.15.x（建议 1.15.0）后重试。"
    )


def _check_local_rknn_onnx_compat():
    try:
        import onnx  # noqa: PLC0415
    except Exception as e:
        raise RuntimeError(f"无法导入 onnx: {e}") from e
    version_text = str(getattr(onnx, "__version__", "") or "")
    if not hasattr(onnx, "mapping"):
        raise RuntimeError(_rknn_onnx_compat_message(version_text))
    return version_text


def _is_image_file(path_obj):
    return str(path_obj.suffix or "").lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_dataset_yaml_root(yaml_path, yaml_data):
    yaml_file = Path(str(yaml_path)).resolve()
    base_dir = yaml_file.parent
    data_path = yaml_data.get("path") if isinstance(yaml_data, dict) else None
    if data_path:
        root = Path(str(data_path)).expanduser()
        if not root.is_absolute():
            root = (base_dir / root).resolve()
        else:
            root = root.resolve()
        return root
    return base_dir


def _collect_dataset_images_from_entry(dataset_root, entry):
    files = []
    if entry in (None, ""):
        return files
    values = entry if isinstance(entry, list) else [entry]
    for value in values:
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = (dataset_root / p).resolve()
        else:
            p = p.resolve()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and _is_image_file(child):
                    files.append(child.resolve())
        elif p.is_file():
            if _is_image_file(p):
                files.append(p.resolve())
            else:
                for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = str(raw or "").strip()
                    if not s or s.startswith("#"):
                        continue
                    child = Path(s).expanduser()
                    if not child.is_absolute():
                        child = (p.parent / child).resolve()
                    else:
                        child = child.resolve()
                    if child.exists() and child.is_file() and _is_image_file(child):
                        files.append(child.resolve())
    return files


def _generate_rknn_dataset_txt(yaml_path, split="train"):
    yaml_file = Path(str(yaml_path)).expanduser()
    if not yaml_file.is_absolute():
        yaml_file = (Path(CTX.CURRENT_DIR) / yaml_file).resolve()
    else:
        yaml_file = yaml_file.resolve()
    if not yaml_file.exists() or not yaml_file.is_file():
        raise RuntimeError("数据集 yaml 文件不存在")
    try:
        yaml_file.relative_to(Path(CTX.CURRENT_DIR).resolve())
    except Exception:
        raise RuntimeError("数据集 yaml 必须位于项目目录内")

    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8", errors="ignore")) or {}
    if not isinstance(data, dict):
        raise RuntimeError("数据集 yaml 内容无效")
    dataset_root = _resolve_dataset_yaml_root(yaml_file, data)

    split_name = str(split or "train").strip().lower()
    requested_entries = []
    if split_name in ("train", "val"):
        requested_entries.append(data.get(split_name))
    elif split_name == "all":
        requested_entries.extend([data.get("train"), data.get("val")])
    else:
        requested_entries.append(data.get("train"))

    image_files = []
    for entry in requested_entries:
        image_files.extend(_collect_dataset_images_from_entry(dataset_root, entry))
    image_files = sorted({str(p): p for p in image_files}.values(), key=lambda p: str(p).lower())
    if not image_files:
        raise RuntimeError("数据集内没有找到可用图片")

    out_path = (yaml_file.parent / "rknn_dataset_auto.txt").resolve()
    lines = []
    for img in image_files:
        try:
            rel = img.relative_to(out_path.parent)
        except Exception:
            continue
        lines.append(rel.as_posix())
    if not lines:
        raise RuntimeError("无法生成相对路径 dataset.txt")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, len(lines)


def _normalize_bmodel_calibration_count(value, default=100):
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = int(default)
    return min(2000, max(1, n))


def _resolve_existing_dir(path_value, field_name):
    p = Path(str(path_value or "")).expanduser()
    if not str(p):
        raise RuntimeError(f"缺少 {field_name}")
    if not p.is_absolute():
        p = (Path(CTX.CURRENT_DIR) / p).resolve()
    else:
        p = p.resolve()
    if not p.exists() or not p.is_dir():
        raise RuntimeError(f"{field_name}目录不存在")
    return p


def _collect_image_files_from_directory(dir_path):
    root = _resolve_existing_dir(dir_path, "标定数据")
    files = []
    for child in sorted(root.rglob("*")):
        try:
            if child.is_file() and _is_image_file(child):
                files.append(child.resolve())
        except Exception:
            continue
    files = sorted({str(p): p for p in files}.values(), key=lambda p: str(p).lower())
    if not files:
        raise RuntimeError("标定数据目录内没有可用图片")
    return files


def _copy_images_to_bmodel_calibration_dir(image_files, out_dir, sample_count=100):
    selected = list(image_files or [])
    if not selected:
        raise RuntimeError("没有可用的标定图片")
    limit = min(len(selected), _normalize_bmodel_calibration_count(sample_count, 100))

    out_dir = Path(str(out_dir)).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for idx, src in enumerate(selected[:limit], start=1):
        src = Path(str(src)).resolve()
        suffix = str(src.suffix or "").lower() or ".jpg"
        dst = (out_dir / f"{idx:04d}_{src.stem}{suffix}").resolve()
        shutil.copy2(str(src), str(dst))
        copied.append(dst)
    if not copied:
        raise RuntimeError("标定数据复制失败")
    return out_dir, len(copied)


def _prepare_bmodel_calibration_dir_from_yaml(yaml_path, split="train", sample_count=100):
    yaml_file = Path(str(yaml_path)).expanduser()
    if not yaml_file.is_absolute():
        yaml_file = (Path(CTX.CURRENT_DIR) / yaml_file).resolve()
    else:
        yaml_file = yaml_file.resolve()
    if not yaml_file.exists() or not yaml_file.is_file():
        raise RuntimeError("数据集 yaml 文件不存在")
    try:
        yaml_file.relative_to(Path(CTX.CURRENT_DIR).resolve())
    except Exception:
        raise RuntimeError("数据集 yaml 必须位于项目目录内")

    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8", errors="ignore")) or {}
    if not isinstance(data, dict):
        raise RuntimeError("数据集 yaml 内容无效")
    dataset_root = _resolve_dataset_yaml_root(yaml_file, data)

    split_name = str(split or "train").strip().lower() or "train"
    requested_entries = []
    if split_name in ("train", "val"):
        requested_entries.append(data.get(split_name))
    elif split_name == "all":
        requested_entries.extend([data.get("train"), data.get("val")])
    else:
        requested_entries.append(data.get("train"))

    image_files = []
    for entry in requested_entries:
        image_files.extend(_collect_dataset_images_from_entry(dataset_root, entry))
    image_files = sorted({str(p): p for p in image_files}.values(), key=lambda p: str(p).lower())
    if not image_files:
        raise RuntimeError("数据集内没有找到可用图片")

    safe_name = re.sub(r"[^0-9a-zA-Z_-]+", "_", yaml_file.stem).strip("_") or "dataset"
    out_dir = (
        Path(CTX.CURRENT_DIR)
        / str(CTX.RESULTS_FOLDER)
        / "bmodel_calibration_auto"
        / f"{safe_name}_{split_name}_{_normalize_bmodel_calibration_count(sample_count, 100)}"
    ).resolve()
    prepared_dir, count = _copy_images_to_bmodel_calibration_dir(image_files, out_dir, sample_count=sample_count)
    return prepared_dir, count


def _prepare_bmodel_calibration_dir_from_source(calibration_dataset, out_dir, sample_count=100):
    image_files = _collect_image_files_from_directory(calibration_dataset)
    return _copy_images_to_bmodel_calibration_dir(image_files, out_dir, sample_count=sample_count)

def _remote_rknn_enabled():
    host = str(os.environ.get("YOLOV11_RKNN_SSH_HOST", "")).strip()
    user = str(os.environ.get("YOLOV11_RKNN_SSH_USER", "")).strip()
    if host and user:
        return True
    try:
        s = CTX.load_settings() or {}
        r = (s.get("rknn_ssh") or {}) if isinstance(s, dict) else {}
        host = str(r.get("host") or "").strip()
        user = str(r.get("user") or "").strip()
        return bool(host and user)
    except Exception:
        return False


def _effective_rknn_ssh():
    # Env overrides settings.json.
    try:
        s = CTX.load_settings() or {}
        r = (s.get("rknn_ssh") or {}) if isinstance(s, dict) else {}
    except Exception:
        r = {}

    def pick(env_name, key, default=""):
        v = os.environ.get(env_name)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
        return str(r.get(key) or default).strip()

    def pick_bool(env_name, key, default=False):
        v = os.environ.get(env_name)
        if v is not None and str(v).strip() != "":
            return str(v).strip().lower() in ("1", "true", "yes")
        try:
            return bool(r.get(key)) if key in r else bool(default)
        except Exception:
            return bool(default)

    return {
        "host": pick("YOLOV11_RKNN_SSH_HOST", "host", ""),
        "user": pick("YOLOV11_RKNN_SSH_USER", "user", ""),
        "port": pick("YOLOV11_RKNN_SSH_PORT", "port", ""),
        "key": pick("YOLOV11_RKNN_SSH_KEY", "key", ""),
        "password": pick("YOLOV11_RKNN_SSH_PASSWORD", "password", ""),
        "remote_base": pick("YOLOV11_RKNN_REMOTE_BASE", "remote_base", "/tmp/yolov11_rknn"),
        "python": pick("YOLOV11_RKNN_PYTHON", "python", "python3"),
        "keep_remote": pick_bool("YOLOV11_KEEP_RKNN_REMOTE", "keep_remote", False),
    }


def _rknn_ssh_with_override(override=None):
    cfg = dict(_effective_rknn_ssh())
    if isinstance(override, dict):
        for k in ["host", "user", "port", "key", "password", "remote_base", "python"]:
            if k in override and override.get(k) is not None:
                cfg[k] = str(override.get(k) or "").strip()
    return cfg


def _sshpass_prefix(password, key):
    pwd = str(password or "").strip()
    if not pwd:
        return []
    if str(key or "").strip():
        # Prefer key auth when key is provided.
        return []
    sshpass = shutil.which("sshpass")
    if not sshpass:
        raise RuntimeError("检测到需要密码登录，但当前环境未安装 sshpass。请改用私钥，或安装 sshpass 后重试。")
    return [str(sshpass), "-p", pwd]


def _ssh_base_args_rknn_from(cfg):
    host = str(cfg.get("host") or "").strip()
    user = str(cfg.get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("缺少 host/user")
    port = str(cfg.get("port") or "").strip()
    key = str(cfg.get("key") or "").strip()
    password = str(cfg.get("password") or "").strip()
    args = _sshpass_prefix(password, key) + ["ssh"]
    if port:
        args += ["-p", port]
    if key:
        args += ["-i", key]
    args += [f"{user}@{host}"]
    return args


def _summarize_command_output(stdout="", stderr="", code=None, max_chars=1800, max_lines=18):
    text = str(stderr or "").strip() or str(stdout or "").strip()
    if not text:
        return f"exit={code}" if code is not None else "命令执行失败"

    cleaned = str(text).replace("\r", "\n").replace("\x08", "")
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", cleaned)

    lines = []
    last_line = None
    for raw in cleaned.splitlines():
        line = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not line:
            continue
        if len(line) > 20 and not re.search(r"[A-Za-z\u4e00-\u9fff]", line):
            continue
        if line == last_line:
            continue
        lines.append(line)
        last_line = line

    if not lines:
        tail = cleaned.strip()
        return ("...\n" + tail[-max_chars:]) if len(tail) > max_chars else tail

    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    msg = "\n".join(lines)
    if len(msg) > max_chars:
        msg = "...\n" + msg[-max_chars:]
    return msg


def _run_subprocess(argv, timeout=None):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        msg = _summarize_command_output(p.stdout, p.stderr, code=p.returncode)
        raise RuntimeError(msg)
    return p.stdout


def _use_password_auth(cfg):
    try:
        return bool(str((cfg or {}).get("password") or "").strip()) and not bool(str((cfg or {}).get("key") or "").strip())
    except Exception:
        return False


def _build_ssh_target(cfg):
    host = str((cfg or {}).get("host") or "").strip()
    user = str((cfg or {}).get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("缺少 host/user")
    return host, user


def _connect_paramiko(cfg):
    try:
        import paramiko  # type: ignore
    except Exception:
        raise RuntimeError("密码登录需要 paramiko。请先安装：pip install paramiko")

    host, user = _build_ssh_target(cfg)
    port = int(str((cfg or {}).get("port") or "22").strip() or 22)
    key = str((cfg or {}).get("key") or "").strip()
    password = str((cfg or {}).get("password") or "").strip() or None

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    kwargs = {
        "hostname": host,
        "port": port,
        "username": user,
        "timeout": 20,
        "look_for_keys": False,
        "allow_agent": False,
    }
    if key:
        kwargs["key_filename"] = key
    if password:
        kwargs["password"] = password
    try:
        client.connect(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "SSH 主机密钥校验失败或连接被拒绝。请先将远端主机指纹加入 known_hosts。"
        ) from exc
    return client


def _remote_exec(cfg, cmd, timeout=60):
    if _use_password_auth(cfg):
        client = _connect_paramiko(cfg)
        try:
            stdin, stdout, stderr = client.exec_command(str(cmd), timeout=timeout)
            code = stdout.channel.recv_exit_status()
            out = (stdout.read() or b"").decode("utf-8", errors="ignore").strip()
            err = (stderr.read() or b"").decode("utf-8", errors="ignore").strip()
            if code != 0:
                msg = _summarize_command_output(out, err, code=code)
                raise RuntimeError(msg)
            return out
        finally:
            try:
                client.close()
            except Exception:
                pass

    # CLI ssh path (key auth or agent auth)
    host, user = _build_ssh_target(cfg)
    port = str((cfg or {}).get("port") or "").strip()
    key = str((cfg or {}).get("key") or "").strip()
    argv = ["ssh"]
    if port:
        argv += ["-p", port]
    if key:
        argv += ["-i", key]
    argv += [f"{user}@{host}", "bash", "-lc", str(cmd)]
    return _run_subprocess(argv, timeout=timeout)


def _remote_upload(cfg, local_path, remote_path, timeout=600):
    local = str(local_path)
    remote = str(remote_path)
    if _use_password_auth(cfg):
        client = _connect_paramiko(cfg)
        try:
            sftp = client.open_sftp()
            try:
                sftp.put(local, remote)
            finally:
                sftp.close()
            return
        finally:
            try:
                client.close()
            except Exception:
                pass

    host, user = _build_ssh_target(cfg)
    port = str((cfg or {}).get("port") or "").strip()
    key = str((cfg or {}).get("key") or "").strip()
    argv = ["scp"]
    if port:
        argv += ["-P", port]
    if key:
        argv += ["-i", key]
    argv += [local, f"{user}@{host}:{remote}"]
    _run_subprocess(argv, timeout=timeout)


def _remote_download(cfg, remote_path, local_path, timeout=600):
    remote = str(remote_path)
    local = str(local_path)
    if _use_password_auth(cfg):
        client = _connect_paramiko(cfg)
        try:
            sftp = client.open_sftp()
            try:
                sftp.get(remote, local)
            finally:
                sftp.close()
            return
        finally:
            try:
                client.close()
            except Exception:
                pass

    host, user = _build_ssh_target(cfg)
    port = str((cfg or {}).get("port") or "").strip()
    key = str((cfg or {}).get("key") or "").strip()
    argv = ["scp"]
    if port:
        argv += ["-P", port]
    if key:
        argv += ["-i", key]
    argv += [f"{user}@{host}:{remote}", local]
    _run_subprocess(argv, timeout=timeout)


def _bmodel_tool_path(env_name, default_name):
    env_v = str(os.environ.get(env_name, "")).strip()
    if env_v:
        return env_v
    found = shutil.which(default_name)
    return found or default_name


def _bmodel_calibration_tool_path(env_name="YOLOV11_BMODEL_CALIBRATION", default_name="run_calibration.py"):
    return _bmodel_tool_path(env_name, default_name)


def _bmodel_tools_available():
    transform = _bmodel_tool_path("YOLOV11_BMODEL_TRANSFORM", "model_transform.py")
    deploy = _bmodel_tool_path("YOLOV11_BMODEL_DEPLOY", "model_deploy.py")
    transform_ok = bool(shutil.which(transform) or Path(str(transform)).exists())
    deploy_ok = bool(shutil.which(deploy) or Path(str(deploy)).exists())
    return transform_ok and deploy_ok


def _detect_local_bmodel_deploy_chip_flag(deploy_tool):
    tool = str(deploy_tool or "").strip()
    if not tool:
        return "--chip"
    try:
        p = subprocess.run([tool, "-h"], capture_output=True, text=True, timeout=30)
        help_text = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    except Exception:
        help_text = ""
    if "--chip" in help_text:
        return "--chip"
    if "--processor" in help_text:
        return "--processor"
    return "--chip"


def _remote_bmodel_enabled():
    host = str(os.environ.get("YOLOV11_BMODEL_SSH_HOST", "")).strip()
    user = str(os.environ.get("YOLOV11_BMODEL_SSH_USER", "")).strip()
    if host and user:
        return True
    try:
        s = CTX.load_settings() or {}
        b = (s.get("bmodel_ssh") or {}) if isinstance(s, dict) else {}
        host = str(b.get("host") or "").strip()
        user = str(b.get("user") or "").strip()
        return bool(host and user)
    except Exception:
        return False


def _effective_bmodel_ssh():
    try:
        s = CTX.load_settings() or {}
        b = (s.get("bmodel_ssh") or {}) if isinstance(s, dict) else {}
    except Exception:
        b = {}

    def pick(env_name, key, default=""):
        v = os.environ.get(env_name)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
        return str(b.get(key) or default).strip()

    def pick_bool(env_name, key, default=False):
        v = os.environ.get(env_name)
        if v is not None and str(v).strip() != "":
            return str(v).strip().lower() in ("1", "true", "yes")
        try:
            return bool(b.get(key)) if key in b else bool(default)
        except Exception:
            return bool(default)

    return {
        "host": pick("YOLOV11_BMODEL_SSH_HOST", "host", ""),
        "user": pick("YOLOV11_BMODEL_SSH_USER", "user", ""),
        "port": pick("YOLOV11_BMODEL_SSH_PORT", "port", ""),
        "key": pick("YOLOV11_BMODEL_SSH_KEY", "key", ""),
        "password": pick("YOLOV11_BMODEL_SSH_PASSWORD", "password", ""),
        "remote_base": pick("YOLOV11_BMODEL_REMOTE_BASE", "remote_base", "/tmp/yolov11_bmodel"),
        "python": pick("YOLOV11_BMODEL_PYTHON", "python", "python3"),
        "transform": pick("YOLOV11_BMODEL_REMOTE_TRANSFORM", "transform", "model_transform.py"),
        "deploy": pick("YOLOV11_BMODEL_REMOTE_DEPLOY", "deploy", "model_deploy.py"),
        "calibration": pick("YOLOV11_BMODEL_REMOTE_CALIBRATION", "calibration", "run_calibration.py"),
        "keep_remote": pick_bool("YOLOV11_KEEP_BMODEL_REMOTE", "keep_remote", False),
    }


def _ssh_base_args_bmodel():
    cfg = _effective_bmodel_ssh()
    host = str(cfg.get("host") or "").strip()
    user = str(cfg.get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("未配置远程BModel转换：请设置环境变量 YOLOV11_BMODEL_SSH_HOST/USER，或在UI「模型转换」页填写BModel远程转换参数")
    port = str(cfg.get("port") or "").strip()
    key = str(cfg.get("key") or "").strip()
    password = str(cfg.get("password") or "").strip()
    args = _sshpass_prefix(password, key) + ["ssh"]
    if port:
        args += ["-p", port]
    if key:
        args += ["-i", key]
    args += [f"{user}@{host}"]
    return args


def _scp_base_args_bmodel():
    cfg = _effective_bmodel_ssh()
    host = str(cfg.get("host") or "").strip()
    user = str(cfg.get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("未配置远程BModel转换：请设置环境变量 YOLOV11_BMODEL_SSH_HOST/USER，或在UI「模型转换」页填写BModel远程转换参数")
    port = str(cfg.get("port") or "").strip()
    key = str(cfg.get("key") or "").strip()
    password = str(cfg.get("password") or "").strip()
    args = _sshpass_prefix(password, key) + ["scp"]
    if port:
        args += ["-P", port]
    if key:
        args += ["-i", key]
    return args, f"{user}@{host}"


def _bmodel_ssh_with_override(override=None):
    cfg = dict(_effective_bmodel_ssh())
    if isinstance(override, dict):
        for k in ["host", "user", "port", "key", "password", "remote_base", "python", "transform", "deploy", "calibration"]:
            if k in override and override.get(k) is not None:
                cfg[k] = str(override.get(k) or "").strip()
    return cfg


def _ssh_base_args_bmodel_from(cfg):
    host = str(cfg.get("host") or "").strip()
    user = str(cfg.get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("缺少 host/user")
    port = str(cfg.get("port") or "").strip()
    key = str(cfg.get("key") or "").strip()
    password = str(cfg.get("password") or "").strip()
    args = _sshpass_prefix(password, key) + ["ssh"]
    if port:
        args += ["-p", port]
    if key:
        args += ["-i", key]
    args += [f"{user}@{host}"]
    return args

def _ssh_base_args():
    cfg = _effective_rknn_ssh()
    host = str(cfg.get("host") or "").strip()
    user = str(cfg.get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("未配置远程RKNN转换：请设置环境变量 YOLOV11_RKNN_SSH_HOST/USER，或在UI「模型转换」页填写RKNN远程转换参数")
    port = str(cfg.get("port") or "").strip()
    key = str(cfg.get("key") or "").strip()
    password = str(cfg.get("password") or "").strip()
    args = _sshpass_prefix(password, key) + ["ssh"]
    if port:
        args += ["-p", port]
    if key:
        args += ["-i", key]
    args += [f"{user}@{host}"]
    return args

def _scp_base_args():
    cfg = _effective_rknn_ssh()
    host = str(cfg.get("host") or "").strip()
    user = str(cfg.get("user") or "").strip()
    if not host or not user:
        raise RuntimeError("未配置远程RKNN转换：请设置环境变量 YOLOV11_RKNN_SSH_HOST/USER，或在UI「模型转换」页填写RKNN远程转换参数")
    port = str(cfg.get("port") or "").strip()
    key = str(cfg.get("key") or "").strip()
    password = str(cfg.get("password") or "").strip()
    args = _sshpass_prefix(password, key) + ["scp"]
    if port:
        args += ["-P", port]
    if key:
        args += ["-i", key]
    return args, f"{user}@{host}"

def _remote_rknn_convert(onnx_path, out_path, platform, do_quant, dataset_txt_path):
    cfg = _effective_rknn_ssh()
    remote_base = str(cfg.get("remote_base") or "/tmp/yolov11_rknn").strip() or "/tmp/yolov11_rknn"
    remote_python = str(cfg.get("python") or "python3").strip() or "python3"
    keep_remote = bool(cfg.get("keep_remote"))

    run_id = uuid.uuid4().hex
    remote_dir = f"{remote_base.rstrip('/')}/{run_id}"
    remote_onnx = f"{remote_dir}/model.onnx"
    remote_rknn = f"{remote_dir}/out.rknn"
    remote_dataset = f"{remote_dir}/dataset.txt"
    remote_script = f"{remote_dir}/convert.py"

    _remote_exec(cfg, f"mkdir -p {shlex.quote(remote_dir)}", timeout=60)
    _remote_upload(cfg, str(Path(onnx_path)), remote_onnx, timeout=600)

    onnx_check_cmd = (
        f"{shlex.quote(remote_python)} -c "
        "\"import onnx; import sys; "
        "v=str(getattr(onnx,'__version__','')); "
        "print(v); "
        "sys.exit(0 if hasattr(onnx,'mapping') else 3)\""
    )
    try:
        onnx_version = _remote_exec(cfg, onnx_check_cmd, timeout=30).strip().splitlines()[-1].strip()
    except Exception as e:
        raise RuntimeError(f"远端 onnx 兼容性检查失败: {e}")
    compat_probe_cmd = (
        f"{shlex.quote(remote_python)} -c "
        "\"import onnx, sys; sys.exit(0 if hasattr(onnx,'mapping') else 3)\"; echo $?"
    )
    compat_status = _remote_exec(cfg, compat_probe_cmd, timeout=30).strip().splitlines()[-1].strip()
    if compat_status != "0":
        raise RuntimeError("远端" + _rknn_onnx_compat_message(onnx_version))

    tmp_paths = []
    try:
        if do_quant:
            if not dataset_txt_path:
                raise RuntimeError("RKNN INT8 量化需要提供 dataset_path")
            dataset_src = Path(str(dataset_txt_path)).expanduser()
            if not dataset_src.is_absolute():
                dataset_src = (Path(CTX.CURRENT_DIR) / dataset_src).resolve()
            else:
                dataset_src = dataset_src.resolve()
            if not dataset_src.exists() or not dataset_src.is_file():
                raise RuntimeError("dataset_path 文件不存在")
            try:
                dataset_src.relative_to(Path(CTX.CURRENT_DIR).resolve())
            except Exception:
                raise RuntimeError("dataset_path 必须位于项目目录内")

            base_dir = dataset_src.parent.resolve()
            raw_lines = dataset_src.read_text(encoding="utf-8", errors="ignore").splitlines()
            entries = []
            for raw in raw_lines:
                s = str(raw or "").strip()
                if not s or s.startswith("#"):
                    continue
                rel = Path(s)
                if rel.is_absolute() or rel.drive:
                    rel = Path(rel.name)
                src = (base_dir / rel).resolve()
                if not src.exists() or not src.is_file():
                    continue
                try:
                    rel_safe = src.relative_to(base_dir)
                except Exception:
                    rel_safe = Path(src.name)
                entries.append((src, rel_safe.as_posix()))
            if not entries:
                raise RuntimeError("dataset_path 内没有有效图片路径（相对路径应相对 dataset.txt 所在目录）")

            tmp_dir = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_rknn").resolve()
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tar_path = (tmp_dir / f"dataset_{run_id}.tar.gz").resolve()
            tmp_paths.append(tar_path)
            with tarfile.open(str(tar_path), "w:gz") as tf:
                for src, arc in entries:
                    tf.add(str(src), arcname=arc)

            remote_tar = f"{remote_dir}/dataset.tar.gz"
            _remote_upload(cfg, str(tar_path), remote_tar, timeout=1200)
            _remote_exec(
                cfg,
                f"mkdir -p {shlex.quote(remote_dir + '/images')} && tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(remote_dir + '/images')}",
                timeout=1200,
            )

            dataset_lines = [f"{remote_dir}/images/{arc}" for _, arc in entries]
            ds_local = (tmp_dir / f"dataset_remote_{run_id}.txt").resolve()
            tmp_paths.append(ds_local)
            ds_local.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")
            _remote_upload(cfg, str(ds_local), remote_dataset, timeout=300)

        script_dir = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_rknn").resolve()
        script_dir.mkdir(parents=True, exist_ok=True)
        script_local = (script_dir / f"convert_{run_id}.py").resolve()
        tmp_paths.append(script_local)
        script_local.write_text(
            "\n".join(
                [
                    "import argparse",
                    "from rknn.api import RKNN",
                    "",
                    "def main():",
                    "    ap = argparse.ArgumentParser()",
                    "    ap.add_argument('--onnx', required=True)",
                    "    ap.add_argument('--out', required=True)",
                    "    ap.add_argument('--platform', default='rk3588')",
                    "    ap.add_argument('--quant', action='store_true')",
                    "    ap.add_argument('--dataset', default='')",
                    "    args = ap.parse_args()",
                    "    rknn = RKNN(verbose=False)",
                    "    try:",
                    "        rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform=args.platform)",
                    "        ret = rknn.load_onnx(model=args.onnx)",
                    "        if ret != 0: raise SystemExit(ret)",
                    "        ret = rknn.build(do_quantization=bool(args.quant), dataset=(args.dataset or None))",
                    "        if ret != 0: raise SystemExit(ret)",
                    "        ret = rknn.export_rknn(args.out)",
                    "        if ret != 0: raise SystemExit(ret)",
                    "    finally:",
                    "        try: rknn.release()",
                    "        except Exception: pass",
                    "",
                    "if __name__ == '__main__':",
                    "    main()",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _remote_upload(cfg, str(script_local), remote_script, timeout=300)

        cmd = [remote_python, remote_script, "--onnx", remote_onnx, "--out", remote_rknn, "--platform", str(platform)]
        if do_quant:
            cmd += ["--quant", "--dataset", remote_dataset]
        _remote_exec(cfg, " ".join(shlex.quote(str(x)) for x in cmd), timeout=3600)

        out_path_local = Path(str(out_path)).expanduser()
        if not out_path_local.is_absolute():
            out_path_local = (Path(CTX.CURRENT_DIR) / out_path_local).resolve()
        out_path_local.parent.mkdir(parents=True, exist_ok=True)
        _remote_download(cfg, remote_rknn, str(out_path_local), timeout=600)
        return str(out_path_local.resolve())
    finally:
        if not keep_remote:
            try:
                _run_subprocess(ssh + ["rm", "-rf", remote_dir], timeout=120)
            except Exception:
                pass
        for p in tmp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def _extract_metric(row, candidates):
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def _sync_curves_from_results(save_dir, username="", task_type=""):
    results_path = os.path.join(str(save_dir), "results.csv")
    if not os.path.exists(results_path):
        return
    train_curve = []
    val_curve = []
    map_curve = []
    precision_curve = []
    recall_curve = []
    with open(results_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_loss = _extract_metric(row, _csv_metric_candidates(task_type, "train_loss"))
            val_map50 = _extract_metric(row, _csv_metric_candidates(task_type, "map50"))
            val_map = _extract_metric(row, _csv_metric_candidates(task_type, "map"))
            precision = _extract_metric(row, _csv_metric_candidates(task_type, "precision"))
            recall = _extract_metric(row, _csv_metric_candidates(task_type, "recall"))
            if train_loss is not None:
                train_curve.append(train_loss)
            if val_map50 is not None:
                val_curve.append(val_map50)
            if val_map is not None:
                map_curve.append(val_map)
            if precision is not None:
                precision_curve.append(precision)
            if recall is not None:
                recall_curve.append(recall)
    with training_status_lock:
        training_status = _get_training_status_ref(username)
        if training_status is None:
            return
        if train_curve:
            training_status["train_curve"] = train_curve
            training_status["loss"] = train_curve[-1]
        if val_curve:
            training_status["val_curve"] = val_curve
            training_status["val_loss"] = val_curve[-1]
        if map_curve:
            training_status["map_curve"] = map_curve
            training_status["mAP"] = map_curve[-1]
        if precision_curve:
            training_status["precision_curve"] = precision_curve
            training_status["precision"] = precision_curve[-1]
        if recall_curve:
            training_status["recall_curve"] = recall_curve
            training_status["recall"] = recall_curve[-1]
        training_status["metric_scope"] = _preferred_metric_scope(task_type)


def _remote_bmodel_convert(
    onnx_path,
    out_path,
    processor,
    quant_mode,
    calibration_table=None,
    calibration_dataset=None,
    calibration_count=100,
    imgsz=640,
    progress_hook=None,
):
    cfg = _effective_bmodel_ssh()
    remote_base = str(cfg.get("remote_base") or "/tmp/yolov11_bmodel").strip() or "/tmp/yolov11_bmodel"
    remote_python = str(cfg.get("python") or "python3").strip() or "python3"
    remote_transform = str(cfg.get("transform") or "model_transform.py").strip() or "model_transform.py"
    remote_deploy = str(cfg.get("deploy") or "model_deploy.py").strip() or "model_deploy.py"
    remote_calibration = str(cfg.get("calibration") or "run_calibration.py").strip() or "run_calibration.py"
    keep_remote = bool(cfg.get("keep_remote"))

    run_id = uuid.uuid4().hex
    remote_dir = f"{remote_base.rstrip('/')}/{run_id}"
    remote_onnx = f"{remote_dir}/model.onnx"
    remote_mlir = f"{remote_dir}/model.mlir"
    remote_out = f"{remote_dir}/out.bmodel"
    remote_cali = f"{remote_dir}/calibration_table"
    remote_cali_dataset = f"{remote_dir}/calibration_dataset"
    remote_cali_tar = f"{remote_dir}/calibration_dataset.tar"
    local_tmp_dir = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_bmodel_remote" / run_id).resolve()
    local_tmp_dir.mkdir(parents=True, exist_ok=True)

    if callable(progress_hook):
        progress_hook(56, "上传 ONNX 到远端...")
    _remote_exec(cfg, f"mkdir -p {shlex.quote(remote_dir)}", timeout=60)
    _remote_upload(cfg, str(Path(onnx_path)), remote_onnx, timeout=600)

    model_name = f"{Path(str(out_path)).stem}_{run_id[:8]}"

    def _tool_prefix(tool):
        t = str(tool or "").strip()
        if "/" in t or "\\" in t:
            if t.lower().endswith(".py"):
                return f"{shlex.quote(remote_python)} {shlex.quote(t)}"
            return shlex.quote(t)
        return shlex.quote(t)

    def _remote_tool_available(tool):
        t = str(tool or "").strip()
        if not t:
            return False
        if "/" in t or "\\" in t:
            cmd = f"[ -f {shlex.quote(t)} ] && echo OK || echo FAIL"
        else:
            cmd = f"command -v {shlex.quote(t)} >/dev/null 2>&1 && echo OK || echo FAIL"
        try:
            return _remote_exec(cfg, cmd, timeout=20).strip() == "OK"
        except Exception:
            return False

    def _detect_remote_deploy_chip_flag():
        try:
            help_text = _remote_exec(cfg, f"{_tool_prefix(remote_deploy)} -h", timeout=60)
        except Exception as e:
            help_text = str(e)
        if "--chip" in help_text:
            return "--chip"
        if "--processor" in help_text:
            return "--processor"
        return "--chip"

    transform_cmd = (
        f"{_tool_prefix(remote_transform)} "
        f"--model_name {shlex.quote(model_name)} "
        f"--model_def {shlex.quote(remote_onnx)} "
        f"--input_shapes {shlex.quote(f'[[1,3,{int(imgsz)},{int(imgsz)}]]')} "
        f"--mlir {shlex.quote(remote_mlir)}"
    )
    deploy_chip_flag = _detect_remote_deploy_chip_flag()

    calibration_cmd = None
    if str(quant_mode).upper() == "INT8":
        if calibration_table:
            cali = Path(str(calibration_table)).expanduser()
            if not cali.is_absolute():
                cali = (Path(CTX.CURRENT_DIR) / cali).resolve()
            else:
                cali = cali.resolve()
            if not cali.exists() or not cali.is_file():
                raise RuntimeError("calibration_table 文件不存在")
            _remote_upload(cfg, str(cali), remote_cali, timeout=600)
        elif calibration_dataset:
            if not _remote_tool_available(remote_calibration):
                raise RuntimeError(f"远端未找到标定工具: {remote_calibration}")
            prepared_dir, use_count = _prepare_bmodel_calibration_dir_from_source(
                calibration_dataset,
                local_tmp_dir / "calibration_dataset",
                sample_count=calibration_count,
            )
            tar_path = (local_tmp_dir / "calibration_dataset.tar").resolve()
            with tarfile.open(str(tar_path), "w") as tar:
                for child in sorted(prepared_dir.iterdir(), key=lambda p: p.name.lower()):
                    tar.add(str(child), arcname=child.name)
            _remote_upload(cfg, str(tar_path), remote_cali_tar, timeout=1200)
            _remote_exec(
                cfg,
                f"mkdir -p {shlex.quote(remote_cali_dataset)} && tar -xf {shlex.quote(remote_cali_tar)} -C {shlex.quote(remote_cali_dataset)}",
                timeout=600,
            )
            calibration_cmd = (
                f"{_tool_prefix(remote_calibration)} "
                f"{shlex.quote(remote_mlir)} "
                f"--dataset {shlex.quote(remote_cali_dataset)} "
                f"--input_num {int(use_count)} "
                f"-o {shlex.quote(remote_cali)}"
            )
        else:
            raise RuntimeError("BModel INT8 量化需要提供 calibration_table，或选择标定数据目录")

    deploy_cmd = (
        f"{_tool_prefix(remote_deploy)} "
        f"--mlir {shlex.quote(remote_mlir)} "
        f"--quantize {shlex.quote(str(quant_mode).upper())} "
        f"{deploy_chip_flag} {shlex.quote(str(processor))} "
        f"--model {shlex.quote(remote_out)}"
    )
    if str(quant_mode).upper() == "INT8":
        deploy_cmd += f" --calibration_table {shlex.quote(remote_cali)}"

    try:
        try:
            if callable(progress_hook):
                progress_hook(64, "执行远端 model_transform.py...")
            _remote_exec(cfg, transform_cmd, timeout=1800)
            if calibration_cmd:
                if callable(progress_hook):
                    progress_hook(74, "执行远端 run_calibration.py...")
                _remote_exec(cfg, calibration_cmd, timeout=3600)
            if callable(progress_hook):
                progress_hook(86, "执行远端 model_deploy.py...")
            _remote_exec(cfg, deploy_cmd, timeout=3600)
        except Exception as e:
            msg = str(e)
            if "No module named 'onnxsim'" in msg or 'No module named "onnxsim"' in msg:
                raise RuntimeError(
                    f"远端 BModel Python 环境缺少 onnxsim，请先在远端执行：{remote_python} -m pip install onnxsim"
                ) from e
            if "run_calibration.py" in msg and ("not found" in msg.lower() or "No such file" in msg):
                raise RuntimeError(f"远端未找到标定工具: {remote_calibration}") from e
            raise

        out_path_local = Path(str(out_path)).expanduser()
        if not out_path_local.is_absolute():
            out_path_local = (Path(CTX.CURRENT_DIR) / out_path_local).resolve()
        else:
            out_path_local = out_path_local.resolve()
        out_path_local.parent.mkdir(parents=True, exist_ok=True)
        if callable(progress_hook):
            progress_hook(94, "下载远端转换结果...")
        _remote_download(cfg, remote_out, str(out_path_local), timeout=600)
        if not out_path_local.exists() or not out_path_local.is_file():
            raise RuntimeError("远程BModel转换完成但本地未收到输出文件")
        return str(out_path_local.resolve())
    finally:
        try:
            shutil.rmtree(local_tmp_dir, ignore_errors=True)
        except Exception:
            pass
        if not keep_remote:
            try:
                _remote_exec(cfg, f"rm -rf {shlex.quote(remote_dir)}", timeout=60)
            except Exception:
                pass


training_history_lock = threading.Lock()
validation_history_lock = threading.Lock()
conversion_history_lock = threading.Lock()

training_history_cache = {}
validation_history_cache = {}
conversion_history_cache = {}


def _history_key(path):
    text = str(path or "")
    return text.lower() if os.name == "nt" else text


def _history_spec(kind):
    kind = str(kind or "").strip().lower()
    if kind == "training":
        return training_history_lock, training_history_cache, "training_history.json"
    if kind == "validation":
        return validation_history_lock, validation_history_cache, "validation_history.json"
    if kind == "conversion":
        return conversion_history_lock, conversion_history_cache, "conversion_history.json"
    raise ValueError(f"未知历史类型: {kind}")


def _history_path(kind):
    _, _, filename = _history_spec(kind)
    root = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER)).resolve()
    return (root / filename).resolve()


def _load_history(path):
    try:
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data[-200:]
    except Exception:
        pass
    return []


def _save_history(path, items):
    try:
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(items or [])[-200:], f, ensure_ascii=False)
    except Exception:
        pass


def _get_history_context_unlocked(kind, reload=False):
    _, cache, _ = _history_spec(kind)
    path = _history_path(kind)
    key = _history_key(path)
    if reload or key not in cache:
        cache[key] = _load_history(path)
    return cache[key], path, key


def _get_history_context(kind, reload=False):
    lock, _, _ = _history_spec(kind)
    with lock:
        return _get_history_context_unlocked(kind, reload=reload)


def _ensure_history_loaded():
    for kind in ("training", "validation", "conversion"):
        _get_history_context(kind, reload=True)


def _append_history_record(kind, record):
    lock, _, _ = _history_spec(kind)
    with lock:
        items, path, _ = _get_history_context_unlocked(kind)
        items.append(record)
        if len(items) > 200:
            items[:] = items[-200:]
        _save_history(path, items)


def _update_history_record(kind, record_id, patch):
    if not record_id:
        return False
    lock, _, _ = _history_spec(kind)
    with lock:
        items, path, _ = _get_history_context_unlocked(kind)
        for rec in items:
            if isinstance(rec, dict) and rec.get("id") == record_id:
                rec.update(dict(patch or {}))
                _save_history(path, items)
                return True
    return False


def _list_owned_history(kind):
    lock, _, _ = _history_spec(kind)
    with lock:
        items, _, _ = _get_history_context_unlocked(kind)
        return [rec for rec in items[-200:] if _record_owned_by_current_user(rec)]


def _pop_owned_history_record(kind, record_id):
    if not record_id:
        return None
    lock, _, _ = _history_spec(kind)
    with lock:
        items, path, _ = _get_history_context_unlocked(kind)
        idx = None
        for i, rec in enumerate(items):
            if isinstance(rec, dict) and rec.get("id") == record_id and _record_owned_by_current_user(rec):
                idx = i
                break
        if idx is None:
            return None
        rec = items.pop(idx)
        _save_history(path, items)
        return rec


def _read_args_yaml_value(save_dir, key):
    try:
        p = Path(str(save_dir)) / "args.yaml"
        if not p.exists() or not p.is_file():
            return None
        prefix = f"{str(key).strip().lower()}:"
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if not s.lower().startswith(prefix):
                    continue
                val = s.split(":", 1)[1].strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                return val.strip()
        return None
    except Exception:
        return None


def _resolve_output_target(output_path, model_name):
    if output_path:
        raw = str(output_path).strip()
        if re.fullmatch(r"^[a-zA-Z]:$", raw):
            raw = ""
        output_target = Path(raw).expanduser() if raw else None
        if output_target is not None:
            output_target = _ensure_path_under_current_dir(output_target, "output_path")
            project_dir = output_target.parent
            run_name = output_target.name or model_name
        else:
            project_dir = (Path(CTX.CURRENT_DIR) / str(CTX.MODELS_FOLDER)).resolve()
            run_name = model_name
    else:
        project_dir = (Path(CTX.CURRENT_DIR) / str(CTX.MODELS_FOLDER)).resolve()
        run_name = model_name
    os.makedirs(str(project_dir), exist_ok=True)
    return str(project_dir), run_name


def _sanitize_model_name_for_filename(model_name):
    text = str(model_name or "").strip()
    if not text:
        return "model"
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        text = text.replace(ch, "_")
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    text = text.strip(" .")
    return text if text else "model"


def _rename_best_weights_by_model_name(save_dir, model_name):
    try:
        save_dir_path = Path(str(save_dir)).resolve()
        weights_dir = save_dir_path / "weights"
        best_path = weights_dir / "best.pt"
        if not best_path.exists() or not best_path.is_file():
            return None
        safe_name = _sanitize_model_name_for_filename(model_name)
        if not safe_name.lower().endswith(".pt"):
            safe_name = f"{safe_name}.pt"
        target_path = weights_dir / safe_name
        if target_path.resolve() == best_path.resolve():
            return str(target_path)
        try:
            if target_path.exists():
                target_path.unlink(missing_ok=True)
        except Exception:
            pass
        shutil.move(str(best_path), str(target_path))
        try:
            shutil.copy2(str(target_path), str(best_path))
        except Exception:
            pass
        return str(target_path)
    except Exception:
        return None


def _normalize_device(device_id):
    if device_id in (None, ""):
        return None
    if str(device_id).strip().lower() in ["-1", "cpu"]:
        return "cpu"
    try:
        return int(device_id)
    except (TypeError, ValueError):
        return str(device_id)


def _resolve_effective_device(device_id):
    normalized = _normalize_device(device_id)
    if normalized is None:
        return None
    if normalized == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return normalized
    return "cpu"


def _is_live_annotation_dataset_yaml(yaml_path):
    try:
        target = Path(str(yaml_path)).resolve()
    except Exception:
        return False
    candidates = [
        (Path(CTX.CURRENT_DIR) / str(CTX.ANNOTATIONS_FOLDER) / "data.yaml").resolve(),
        (Path(CTX.CURRENT_DIR) / str(CTX.DATASETS_FOLDER) / _AUTO_DATASET_NAME / "data.yaml").resolve(),
    ]
    return any(target == candidate for candidate in candidates)


def _prepare_training_dataset_yaml(yaml_path, train_split):
    yaml_file = resolve_existing_file(yaml_path, allowed_exts=["yaml", "yml"], current_dir=CTX.CURRENT_DIR)
    yaml_file = Path(str(yaml_file)).resolve()

    if not _is_live_annotation_dataset_yaml(yaml_file):
        _apply_train_split_to_dataset(str(yaml_file), train_split)
        return str(yaml_file), None

    data = yaml.safe_load(yaml_file.read_text(encoding="utf-8", errors="ignore")) or {}
    if not isinstance(data, dict):
        raise RuntimeError("数据集 yaml 内容无效")
    dataset_root = _resolve_dataset_yaml_root(yaml_file, data)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise RuntimeError("数据集目录不存在")

    snapshot_root = (
        Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_train_dataset" / uuid.uuid4().hex
    ).resolve()
    snapshot_dataset_root = (snapshot_root / "dataset").resolve()
    shutil.copytree(str(dataset_root), str(snapshot_dataset_root))

    snapshot_payload = deepcopy(data)
    snapshot_payload["path"] = str(snapshot_dataset_root).replace("\\", "/")
    snapshot_yaml = (snapshot_root / yaml_file.name).resolve()
    snapshot_yaml.write_text(
        yaml.safe_dump(snapshot_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _apply_train_split_to_dataset(str(snapshot_yaml), train_split)
    return str(snapshot_yaml), snapshot_root


def _apply_train_split_to_dataset(yaml_path, train_split):
    if not yaml_path:
        return
    dataset_path = os.path.dirname(os.path.abspath(str(yaml_path)))
    images_root = os.path.join(dataset_path, "images")
    labels_root = os.path.join(dataset_path, "labels")
    train_img_dir = os.path.join(images_root, "train")
    val_img_dir = os.path.join(images_root, "val")
    train_lbl_dir = os.path.join(labels_root, "train")
    val_lbl_dir = os.path.join(labels_root, "val")
    if not (os.path.isdir(train_img_dir) and os.path.isdir(val_img_dir)):
        return

    image_records = []
    for split_name, split_dir in [("train", train_img_dir), ("val", val_img_dir)]:
        for filename in os.listdir(split_dir):
            src_img = os.path.join(split_dir, filename)
            if os.path.isfile(src_img):
                image_records.append((filename, split_name, src_img))
    if len(image_records) < 2:
        return

    image_records.sort(key=lambda x: str(x[0]).lower())
    ratio = _normalize_train_split(train_split) / 100.0
    train_count = int(round(len(image_records) * ratio))
    train_count = min(len(image_records) - 1, max(1, train_count))
    train_names = {name for name, _, _ in image_records[:train_count]}

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    for filename, old_split, src_img in image_records:
        new_split = "train" if filename in train_names else "val"
        if new_split == old_split:
            continue
        dst_img = os.path.join(images_root, new_split, filename)
        try:
            if os.path.exists(dst_img) and os.path.isfile(dst_img):
                os.remove(dst_img)
        except Exception:
            pass
        try:
            shutil.move(src_img, dst_img)
        except Exception:
            continue

        stem = os.path.splitext(filename)[0]
        src_lbl = os.path.join(labels_root, old_split, f"{stem}.txt")
        dst_lbl = os.path.join(labels_root, new_split, f"{stem}.txt")
        if os.path.exists(src_lbl) and os.path.isfile(src_lbl):
            try:
                if os.path.exists(dst_lbl) and os.path.isfile(dst_lbl):
                    os.remove(dst_lbl)
            except Exception:
                pass
            try:
                shutil.move(src_lbl, dst_lbl)
            except Exception:
                pass


def _resolve_plot_path(save_dir_path, candidate_filenames):
    for filename in candidate_filenames:
        file_path = save_dir_path / filename
        if file_path.exists() and file_path.is_file():
            return file_path
    lowered = {name.lower(): name for name in candidate_filenames}
    try:
        for file_path in save_dir_path.iterdir():
            if not file_path.is_file():
                continue
            name_lower = file_path.name.lower()
            if name_lower in lowered:
                return file_path
    except Exception:
        pass
    try:
        for filename in candidate_filenames:
            matches = list(save_dir_path.rglob(filename))
            for match in matches:
                if match.exists() and match.is_file():
                    return match
    except Exception:
        pass
    return None


def train_model_async(
    yaml_path,
    epochs,
    batch_size,
    imgsz,
    workers=4,
    model_name="train",
    pretrained_model="yolo11n.pt",
    device_id=None,
    learning_rate=None,
    train_split=80,
    output_path=None,
    augment=None,
    username="",
    workspace_dir="",
    task_type="",
):
    dataset_snapshot_dir = None
    if workspace_dir:
        try:
            CTX.set_thread_runtime(username=username, workspace_dir=workspace_dir, current_task=task_type)
        except Exception:
            pass
    _ensure_history_loaded()
    try:
        task_type = _normalize_task_type(task_type)
        run_record = {
            "id": uuid.uuid4().hex,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "owner": str(username or ""),
            "task_type": task_type,
            "status": "training",
            "yaml_path": yaml_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "workers": workers,
            "model_name": model_name,
            "pretrained_model": pretrained_model,
            "device_id": device_id,
            "learning_rate": learning_rate,
            "train_split": train_split,
            "output_path": output_path,
            "augment": augment,
        }
        run_record["model_type"] = _infer_model_type(pretrained_model) or _infer_model_type(model_name)

        train_split_norm = _normalize_train_split(train_split)
        with training_status_lock:
            training_status = _get_training_status_ref(username, create=True)
            training_status.clear()
            training_status.update(_empty_training_status())
            training_status["owner"] = str(username or "")
            training_status["status"] = "training"
            training_status["progress"] = 0
            training_status["message"] = "开始训练..."
            training_status["loss"] = None
            training_status["val_loss"] = None
            training_status["mAP"] = None
            training_status["log"] = None
            training_status["train_curve"] = []
            training_status["val_curve"] = []
            training_status["map_curve"] = []
            training_status["train_split"] = train_split_norm
            training_status["task_type"] = task_type
            training_status["metric_scope"] = _preferred_metric_scope(task_type)

        resolved_model_path = None
        if pretrained_model:
            try:
                resolved_model_path = str(_resolve_model_input_path(pretrained_model, allowed_exts=["pt"]))
            except Exception:
                resolved_model_path = None
        if resolved_model_path is None:
            resolved_model_path = os.path.join(str(CTX.MODELS_FOLDER), str(pretrained_model))
        model_path = resolved_model_path

        original_torch_load = torch.load

        def patched_torch_load(f, map_location=None, **kwargs):
            return CTX.torch_load_with_compat(original_torch_load, f, map_location=map_location, **kwargs)

        torch.load = patched_torch_load

        try:
            model = YOLO(model_path)

            batch_state = {"current": 0, "total": 0}

            def on_train_start(trainer):
                with training_status_lock:
                    training_status = _get_training_status_ref(username, create=True)
                    training_status["status"] = "training"
                    training_status["progress"] = 0
                    training_status["message"] = "开始训练..."
                    training_status["owner"] = str(username or "")
                    training_status["current_batch"] = 0
                    training_status["total_batches"] = 0

            def on_train_epoch_start(trainer):
                batch_state["current"] = 0
                try:
                    batch_state["total"] = len(trainer.train_loader) if hasattr(trainer, "train_loader") else 0
                except Exception:
                    batch_state["total"] = 0
                with training_status_lock:
                    training_status = _get_training_status_ref(username, create=True)
                    training_status["current_batch"] = 0
                    training_status["total_batches"] = batch_state["total"]

            def on_train_batch_end(trainer):
                batch_state["current"] += 1
                total_batches = batch_state["total"] if batch_state["total"] > 0 else 1
                epoch_index = trainer.epoch if hasattr(trainer, "epoch") else 0
                fine_progress = ((epoch_index + (batch_state["current"] / total_batches)) / max(epochs, 1)) * 100
                with training_status_lock:
                    training_status = _get_training_status_ref(username, create=True)
                    training_status["progress"] = min(99, max(training_status.get("progress", 0), int(fine_progress)))
                    training_status["current_batch"] = batch_state["current"]
                    training_status["total_batches"] = batch_state["total"]
                    training_status["message"] = f"Epoch {epoch_index + 1}/{epochs} - Batch {batch_state['current']}/{batch_state['total']}"

            def on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                loss_value = _extract_training_loss_value(trainer, task_type)
                log_msg = f"Epoch {epoch}/{epochs}"
                with training_status_lock:
                    training_status = _get_training_status_ref(username, create=True)
                    training_status["progress"] = int((epoch / max(epochs, 1)) * 100)
                    training_status["message"] = f"Epoch {epoch}/{epochs}"
                    if loss_value is not None:
                        training_status["loss"] = loss_value
                    if training_status.get("loss") is not None:
                        log_msg += f" - loss: {float(training_status['loss']):.4f}"
                    training_status["log"] = log_msg

            def on_fit_epoch_end(trainer):
                epoch = trainer.epoch + 1
                map_value = None
                map50_value = None
                _, metric_scope = None, _preferred_metric_scope(task_type)
                metrics_obj, metric_scope = _resolve_results_metric_container(getattr(trainer, "metrics", None), task_type)
                if metrics_obj is not None:
                    if hasattr(metrics_obj, "map"):
                        try:
                            map_value = float(metrics_obj.map)
                        except Exception:
                            map_value = None
                    if hasattr(metrics_obj, "map50"):
                        try:
                            map50_value = float(metrics_obj.map50)
                        except Exception:
                            map50_value = None
                log_msg = f"Epoch {epoch}/{epochs}"
                with training_status_lock:
                    training_status = _get_training_status_ref(username, create=True)
                    if map_value is not None:
                        training_status["mAP"] = map_value
                    if map50_value is not None:
                        training_status["val_loss"] = map50_value
                    training_status["metric_scope"] = metric_scope
                    if training_status.get("loss") is not None:
                        log_msg += f" - loss: {float(training_status['loss']):.4f}"
                    if training_status.get("mAP") is not None:
                        log_msg += f", {'Mask mAP' if metric_scope == 'mask' else 'mAP'}: {float(training_status['mAP']):.4f}"
                    training_status["log"] = log_msg
                if hasattr(trainer, "save_dir"):
                    _sync_curves_from_results(trainer.save_dir, username=username, task_type=task_type)

            def on_train_end(trainer):
                with training_status_lock:
                    training_status = _get_training_status_ref(username, create=True)
                    training_status["status"] = "completed"
                    training_status["progress"] = 100
                    training_status["message"] = "训练完成！"

            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_batch_end", on_train_batch_end)
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.add_callback("on_train_end", on_train_end)

            effective_workers = workers if isinstance(workers, int) else int(workers)
            if effective_workers < 0:
                effective_workers = 0
            if threading.current_thread() is not threading.main_thread() and effective_workers > 0:
                effective_workers = 0
            if not torch.cuda.is_available() and effective_workers > 0:
                effective_workers = 0
            effective_device = _resolve_effective_device(device_id)
            with training_status_lock:
                training_status = _get_training_status_ref(username, create=True)
                training_status["device"] = (
                    effective_device if effective_device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
                )

            project_dir, run_name = _resolve_output_target(output_path, model_name)
            yaml_path, dataset_snapshot_dir = _prepare_training_dataset_yaml(yaml_path, train_split_norm)
            with training_status_lock:
                training_status = _get_training_status_ref(username, create=True)
                training_status["output_path"] = str(Path(project_dir) / run_name)
                training_status["data_yaml"] = str(yaml_path)

            train_kwargs = dict(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                workers=effective_workers,
                project=project_dir,
                name=run_name,
                exist_ok=True,
            )
            train_kwargs["optimizer"] = "SGD"
            if effective_device is not None:
                train_kwargs["device"] = effective_device
            if learning_rate is not None:
                try:
                    lr = float(learning_rate)
                    if lr > 0:
                        train_kwargs["lr0"] = lr
                except (TypeError, ValueError):
                    pass

            if isinstance(augment, dict):
                enabled = bool(augment.get("enabled", True))
                if not enabled:
                    train_kwargs.update(dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, auto_augment="none", close_mosaic=0))
                else:
                    for key in ["mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "erasing"]:
                        if key in augment:
                            try:
                                train_kwargs[key] = float(augment.get(key))
                            except Exception:
                                pass
                    if "close_mosaic" in augment:
                        try:
                            train_kwargs["close_mosaic"] = int(float(augment.get("close_mosaic")))
                        except Exception:
                            pass
                    auto_aug = str(augment.get("auto_augment") or "").strip().lower()
                    if auto_aug:
                        train_kwargs["auto_augment"] = auto_aug

            model.train(**train_kwargs)

            with training_status_lock:
                training_status = _get_training_status_ref(username, create=True)
                training_status["status"] = "completed"
                training_status["progress"] = 100
                training_status["message"] = "训练完成！"
                snapshot = dict(training_status)

            output_dir = os.path.join(project_dir, run_name)
            _sync_curves_from_results(output_dir, username=username, task_type=task_type)

            renamed_best = _rename_best_weights_by_model_name(snapshot.get("output_path"), model_name)
            run_record["status"] = "completed"
            run_record["message"] = snapshot.get("message")
            run_record["device"] = snapshot.get("device")
            run_record["output_dir"] = snapshot.get("output_path")
            guessed_type = _infer_model_type(_read_args_yaml_value(output_dir, "model")) or _infer_model_type(pretrained_model) or _infer_model_type(model_path)
            if guessed_type:
                run_record["model_type"] = guessed_type
            if renamed_best:
                run_record["best_weights"] = renamed_best
            run_record["metrics"] = {
                "loss": snapshot.get("loss"),
                "mAP": snapshot.get("mAP"),
                "map50": snapshot.get("val_loss"),
                "precision": snapshot.get("precision"),
                "recall": snapshot.get("recall"),
                "metric_scope": snapshot.get("metric_scope") or _preferred_metric_scope(task_type),
            }
            _append_history_record("training", run_record)
        finally:
            torch.load = original_torch_load
    except Exception as e:
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        with training_status_lock:
            training_status = _get_training_status_ref(username, create=True)
            training_status["status"] = "error"
            training_status["message"] = f"训练出错: {str(e)}"
            snapshot = dict(training_status)
        try:
            run_record["status"] = "error"
            run_record["message"] = snapshot.get("message")
            run_record["device"] = snapshot.get("device")
            run_record["output_dir"] = snapshot.get("output_path")
            _append_history_record("training", run_record)
        except Exception:
            pass
    finally:
        if dataset_snapshot_dir:
            try:
                shutil.rmtree(str(dataset_snapshot_dir), ignore_errors=True)
            except Exception:
                pass
        try:
            CTX.clear_thread_runtime()
        except Exception:
            pass


def validate_model_async(model_path, data_path=None, conf=0.25, iou=0.7, username="", workspace_dir="", task_type=""):
    if workspace_dir:
        try:
            CTX.set_thread_runtime(username=username, workspace_dir=workspace_dir, current_task=task_type)
        except Exception:
            pass
    _ensure_history_loaded()
    try:
        task_type = _normalize_task_type(task_type)
        with validation_status_lock:
            validation_status = _get_validation_status_ref(username, create=True)
            validation_status.clear()
            validation_status.update(_empty_validation_status())
            validation_status["owner"] = str(username or "")
            validation_status["task_type"] = task_type
            validation_status["status"] = "validating"
            validation_status["progress"] = 0
            validation_status["message"] = "开始评估..."
            validation_status["model_path"] = model_path
            validation_status["data_path"] = data_path
            validation_status["conf"] = conf
            validation_status["iou"] = iou
            validation_status["plots"] = {}

        original_torch_load = torch.load

        def patched_torch_load(f, map_location=None, **kwargs):
            return CTX.torch_load_with_compat(original_torch_load, f, map_location=map_location, **kwargs)

        torch.load = patched_torch_load
        try:
            model = YOLO(model_path)
            with validation_status_lock:
                validation_status = _get_validation_status_ref(username, create=True)
                validation_status["progress"] = 30
            val_kwargs = dict(conf=conf, iou=iou, plots=True)
            if data_path:
                val_kwargs["data"] = data_path
            results = model.val(**val_kwargs)
            with validation_status_lock:
                validation_status = _get_validation_status_ref(username, create=True)
                validation_status["progress"] = 90
        finally:
            torch.load = original_torch_load

        metrics_obj, metric_scope = _resolve_results_metric_container(results, task_type)
        if metrics_obj is None:
            raise RuntimeError("未获取到有效评估指标")
        results_payload = {
            "mAP50": float(metrics_obj.map50),
            "mAP50_95": float(metrics_obj.map),
            "precision": float(metrics_obj.mp),
            "recall": float(metrics_obj.mr),
            "metric_scope": metric_scope,
            "task_type": task_type,
        }
        save_dir = getattr(results, "save_dir", None)
        plots_payload = {}
        if save_dir:
            save_dir_path = Path(save_dir).resolve()
            current_root = Path(CTX.CURRENT_DIR).resolve()
            try:
                save_dir_path.relative_to(current_root)
                under_root = True
            except Exception:
                under_root = False
            if under_root:
                if metric_scope == "mask":
                    plot_candidates = {
                        "pr": ["MaskPR_curve.png", "SegPR_curve.png", "PR_curve.png", "pr_curve.png"],
                        "f1": ["MaskF1_curve.png", "SegF1_curve.png", "F1_curve.png", "f1_curve.png"],
                    }
                else:
                    plot_candidates = {
                        "pr": ["BoxPR_curve.png", "PR_curve.png", "pr_curve.png"],
                        "f1": ["BoxF1_curve.png", "F1_curve.png", "f1_curve.png"],
                    }
                for key, candidates in plot_candidates.items():
                    file_path = _resolve_plot_path(save_dir_path, candidates)
                    if file_path:
                        rel_file = file_path.relative_to(current_root).as_posix()
                        plots_payload[key] = f"/api/eval-plot/{rel_file}"

        with validation_status_lock:
            validation_status = _get_validation_status_ref(username, create=True)
            validation_status["status"] = "completed"
            validation_status["progress"] = 100
            validation_status["message"] = "评估完成！"
            validation_status["results"] = results_payload
            validation_status["save_dir"] = str(save_dir) if save_dir else None
            validation_status["plots"] = plots_payload

        record = {
            "id": uuid.uuid4().hex,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "owner": str(username or ""),
            "task_type": task_type,
            "model_path": model_path,
            "data_path": data_path,
            "conf": conf,
            "iou": iou,
            "results": results_payload,
            "plots": plots_payload,
            "save_dir": str(save_dir) if save_dir else None,
        }
        _append_history_record("validation", record)
    except Exception as e:
        with validation_status_lock:
            validation_status = _get_validation_status_ref(username, create=True)
            validation_status["status"] = "error"
            validation_status["message"] = f"评估出错: {str(e)}"
    finally:
        try:
            CTX.clear_thread_runtime()
        except Exception:
            pass


def convert_model_async(
    model_path,
    export_format,
    precision="fp32",
    output_path=None,
    imgsz=640,
    target_platform=None,
    dataset_path=None,
    calibration_table=None,
    calibration_dataset=None,
    calibration_count=100,
    username="",
    workspace_dir="",
    task_type="",
):
    temp_export_dirs = []
    if workspace_dir:
        try:
            CTX.set_thread_runtime(username=username, workspace_dir=workspace_dir, current_task=task_type)
        except Exception:
            pass
    _ensure_history_loaded()
    try:
        task_type = _normalize_task_type(task_type)
        with conversion_status_lock:
            conversion_status = _get_conversion_status_ref(username, create=True)
            record_id = conversion_status.get("record_id")
        started_at = datetime.datetime.now().isoformat(timespec="seconds")
        t_record0 = time.perf_counter()
        with conversion_status_lock:
            conversion_status = _get_conversion_status_ref(username, create=True)
            conversion_status.setdefault("owner", str(username or ""))
            conversion_status["task_type"] = task_type
            conversion_status["status"] = "converting"
            conversion_status["progress"] = 0
            conversion_status["message"] = "开始转换..."
            conversion_status["model_path"] = model_path
            conversion_status["format"] = export_format
            conversion_status["precision"] = precision
            conversion_status.pop("output_path", None)
            conversion_status["started_at"] = started_at

        def _prepare_rknn_dataset_file(dataset_txt_path):
            src = Path(str(dataset_txt_path)).expanduser()
            if not src.is_absolute():
                src = (Path(CTX.CURRENT_DIR) / src).resolve()
            else:
                src = src.resolve()
            if not src.exists() or not src.is_file():
                raise RuntimeError("dataset_path 文件不存在")

            try:
                src.relative_to(Path(CTX.CURRENT_DIR).resolve())
            except Exception:
                raise RuntimeError("dataset_path 必须位于项目目录内")

            lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
            abs_paths = []
            base_dir = src.parent
            for raw in lines:
                s = str(raw or "").strip()
                if not s or s.startswith("#"):
                    continue
                p = Path(s).expanduser()
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                else:
                    p = p.resolve()
                if not p.exists() or not p.is_file():
                    continue
                abs_paths.append(str(p))
            if not abs_paths:
                raise RuntimeError("dataset_path 内没有有效图片路径（请检查相对路径是否相对 dataset.txt 所在目录）")

            out_dir = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_rknn").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = (out_dir / f"dataset_{uuid.uuid4().hex}.txt").resolve()
            out_path.write_text("\n".join(abs_paths) + "\n", encoding="utf-8")
            return str(out_path)

        if export_format == "rknn":
            # RKNN takes ONNX as input. If given a .pt, export to ONNX first.
            mp = str(model_path)
            suffix = mp.lower()
            onnx_path = None
            if suffix.endswith(".onnx"):
                onnx_path = str(Path(mp).resolve())
            else:
                with conversion_status_lock:
                    conversion_status = _get_conversion_status_ref(username, create=True)
                    conversion_status["progress"] = 15
                    conversion_status["message"] = "导出ONNX..."
                exported_path = _export_model_for_conversion(
                    mp,
                    {"format": "onnx", "imgsz": imgsz},
                    temp_export_dirs=temp_export_dirs,
                )
                onnx_path = str(exported_path)

            if not onnx_path or not Path(onnx_path).exists():
                raise RuntimeError("ONNX导出失败")
            with conversion_status_lock:
                conversion_status = _get_conversion_status_ref(username, create=True)
                conversion_status["progress"] = 40
                conversion_status["message"] = "构建RKNN..."

            do_quant = (str(precision).lower() == "int8")
            platform = str(target_platform or "rk3588").strip() or "rk3588"

            if CTX.module_available("rknn"):
                try:
                    from rknn.api import RKNN
                except Exception:
                    RKNN = None
                if RKNN is None:
                    raise RuntimeError("rknn 模块存在但导入失败")
                _check_local_rknn_onnx_compat()

                if do_quant and not dataset_path:
                    raise RuntimeError("RKNN INT8 量化需要提供 dataset_path（一个包含图片路径列表的 txt 文件）")

                rknn = RKNN(verbose=False)
                tmp_dataset = None
                try:
                    rknn.config(
                        mean_values=[[0, 0, 0]],
                        std_values=[[255, 255, 255]],
                        target_platform=platform,
                    )
                    ret = rknn.load_onnx(model=str(onnx_path))
                    if ret != 0:
                        raise RuntimeError(f"RKNN load_onnx 失败: {ret}")
                    with conversion_status_lock:
                        conversion_status = _get_conversion_status_ref(username, create=True)
                        conversion_status["progress"] = 65
                    if do_quant:
                        tmp_dataset = _prepare_rknn_dataset_file(dataset_path)
                    ret = rknn.build(do_quantization=bool(do_quant), dataset=str(tmp_dataset) if do_quant else None)
                    if ret != 0:
                        raise RuntimeError(f"RKNN build 失败: {ret}")
                    with conversion_status_lock:
                        conversion_status = _get_conversion_status_ref(username, create=True)
                        conversion_status["progress"] = 85

                    desired_path = _resolve_conversion_output_path(output_path, model_path, "rknn")
                    os.makedirs(str(desired_path.parent), exist_ok=True)
                    ret = rknn.export_rknn(str(desired_path))
                    if ret != 0:
                        raise RuntimeError(f"RKNN export_rknn 失败: {ret}")
                    final_path = desired_path.resolve()
                finally:
                    try:
                        rknn.release()
                    except Exception:
                        pass
                    if tmp_dataset and os.environ.get("YOLOV11_KEEP_RKNN_DATASET", "").strip() not in ("1", "true", "yes"):
                        try:
                            Path(str(tmp_dataset)).unlink(missing_ok=True)
                        except Exception:
                            pass
            else:
                if not _remote_rknn_enabled():
                    raise RuntimeError("缺少 rknn-toolkit2，且未配置远程SSH转换（设置环境变量 YOLOV11_RKNN_SSH_HOST/USER，或在UI「模型转换」页填写）")
                if do_quant and not dataset_path:
                    raise RuntimeError("RKNN INT8 量化需要提供 dataset_path（一个包含图片路径列表的 txt 文件）")
                with conversion_status_lock:
                    conversion_status = _get_conversion_status_ref(username, create=True)
                    conversion_status["progress"] = 55
                    conversion_status["message"] = "上传到远端并转换..."
                desired_path = _resolve_conversion_output_path(output_path, model_path, "rknn")
                final_path = Path(_remote_rknn_convert(onnx_path, desired_path, platform, bool(do_quant), dataset_path)).resolve()
        elif export_format == "bmodel":
            # TPU-MLIR flow: ONNX -> MLIR -> BMODEL
            mp = str(model_path)
            suffix = mp.lower()
            onnx_path = None
            if suffix.endswith(".onnx"):
                onnx_path = str(Path(mp).resolve())
            else:
                with conversion_status_lock:
                    conversion_status = _get_conversion_status_ref(username, create=True)
                    conversion_status["progress"] = 15
                    conversion_status["message"] = "导出ONNX..."
                exported_path = _export_model_for_conversion(
                    mp,
                    {"format": "onnx", "imgsz": imgsz},
                    temp_export_dirs=temp_export_dirs,
                )
                onnx_path = str(exported_path)

            if not onnx_path or not Path(onnx_path).exists():
                raise RuntimeError("ONNX导出失败")

            quant_mode_map = {"fp32": "F32", "fp16": "F16", "int8": "INT8"}
            quant_mode = quant_mode_map.get(str(precision).lower(), "F32")
            processor = str(target_platform or "cv186ah").strip() or "cv186ah"

            desired_path = _resolve_conversion_output_path(output_path, model_path, "bmodel")
            os.makedirs(str(desired_path.parent), exist_ok=True)

            local_bmodel_ready = _bmodel_tools_available()
            local_calibration_tool_ok = True
            if quant_mode == "INT8" and not calibration_table and calibration_dataset:
                calibration_tool = _bmodel_calibration_tool_path("YOLOV11_BMODEL_CALIBRATION", "run_calibration.py")
                local_calibration_tool_ok = bool(shutil.which(str(calibration_tool)) or Path(str(calibration_tool)).exists())
            if local_bmodel_ready and local_calibration_tool_ok:
                transform_tool = _bmodel_tool_path("YOLOV11_BMODEL_TRANSFORM", "model_transform.py")
                deploy_tool = _bmodel_tool_path("YOLOV11_BMODEL_DEPLOY", "model_deploy.py")

                run_id = uuid.uuid4().hex[:8]
                tmp_dir = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_bmodel" / run_id).resolve()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                mlir_path = (tmp_dir / "model.mlir").resolve()

                with conversion_status_lock:
                    conversion_status = _get_conversion_status_ref(username, create=True)
                    conversion_status["progress"] = 40
                    conversion_status["message"] = "执行 model_transform.py..."

                model_name = f"{Path(str(model_path)).stem}_{run_id}"
                transform_cmd = [
                    str(transform_tool),
                    "--model_name",
                    model_name,
                    "--model_def",
                    str(onnx_path),
                    "--input_shapes",
                    f"[[1,3,{int(imgsz)},{int(imgsz)}]]",
                    "--mlir",
                    str(mlir_path),
                ]
                _run_subprocess(transform_cmd, timeout=1800)

                with conversion_status_lock:
                    conversion_status = _get_conversion_status_ref(username, create=True)
                    conversion_status["progress"] = 70
                    conversion_status["message"] = "执行 model_deploy.py..."

                deploy_chip_flag = _detect_local_bmodel_deploy_chip_flag(deploy_tool)
                deploy_cmd = [
                    str(deploy_tool),
                    "--mlir",
                    str(mlir_path),
                    "--quantize",
                    quant_mode,
                    deploy_chip_flag,
                    processor,
                    "--model",
                    str(desired_path),
                ]
                if quant_mode == "INT8":
                    if calibration_table:
                        cali = Path(str(calibration_table)).expanduser()
                        if not cali.is_absolute():
                            cali = (Path(CTX.CURRENT_DIR) / cali).resolve()
                        else:
                            cali = cali.resolve()
                        if not cali.exists() or not cali.is_file():
                            raise RuntimeError("calibration_table 文件不存在")
                    elif calibration_dataset:
                        calibration_tool = _bmodel_calibration_tool_path("YOLOV11_BMODEL_CALIBRATION", "run_calibration.py")
                        calibration_tool_ok = bool(shutil.which(str(calibration_tool)) or Path(str(calibration_tool)).exists())
                        if not calibration_tool_ok:
                            raise RuntimeError("当前环境缺少 run_calibration.py，无法根据标定数据目录自动生成 calibration_table")

                        with conversion_status_lock:
                            conversion_status = _get_conversion_status_ref(username, create=True)
                            conversion_status["progress"] = 58
                            conversion_status["message"] = "准备标定数据..."

                        prepared_calibration_dir, use_count = _prepare_bmodel_calibration_dir_from_source(
                            calibration_dataset,
                            tmp_dir / "calibration_dataset",
                            sample_count=calibration_count,
                        )
                        cali = (tmp_dir / "calibration_table").resolve()

                        with conversion_status_lock:
                            conversion_status = _get_conversion_status_ref(username, create=True)
                            conversion_status["progress"] = 64
                            conversion_status["message"] = "执行 run_calibration.py..."

                        calibration_cmd = [
                            str(calibration_tool),
                            str(mlir_path),
                            "--dataset",
                            str(prepared_calibration_dir),
                            "--input_num",
                            str(int(use_count)),
                            "-o",
                            str(cali),
                        ]
                        _run_subprocess(calibration_cmd, timeout=3600)
                        if not cali.exists() or not cali.is_file():
                            raise RuntimeError("run_calibration.py 执行完成但未生成 calibration_table")
                    else:
                        raise RuntimeError("BModel INT8 量化需要提供 calibration_table，或选择标定数据目录")

                    deploy_cmd += ["--calibration_table", str(cali)]

                _run_subprocess(deploy_cmd, timeout=3600)
                if not desired_path.exists() or not desired_path.is_file():
                    raise RuntimeError("BModel 导出失败：未生成输出文件")
                final_path = desired_path.resolve()
            else:
                if local_bmodel_ready and not local_calibration_tool_ok and not _remote_bmodel_enabled():
                    raise RuntimeError("当前环境缺少 run_calibration.py，无法根据标定数据目录自动生成 calibration_table")
                if not _remote_bmodel_enabled():
                    raise RuntimeError("缺少 tpu-mlir，且未配置远程SSH转换（YOLOV11_BMODEL_SSH_HOST/USER 或UI「模型转换」页）")
                def _set_remote_bmodel_progress(progress, message):
                    with conversion_status_lock:
                        conversion_status = _get_conversion_status_ref(username, create=True)
                        conversion_status["progress"] = int(progress)
                        conversion_status["message"] = str(message or "")

                _set_remote_bmodel_progress(55, "上传到远端并转换...")
                final_path = Path(
                    _remote_bmodel_convert(
                        onnx_path=onnx_path,
                        out_path=desired_path,
                        processor=processor,
                        quant_mode=quant_mode,
                        calibration_table=calibration_table,
                        calibration_dataset=calibration_dataset,
                        calibration_count=calibration_count,
                        imgsz=imgsz,
                        progress_hook=_set_remote_bmodel_progress,
                    )
                ).resolve()
        else:
            with conversion_status_lock:
                conversion_status = _get_conversion_status_ref(username, create=True)
                conversion_status["progress"] = 25

            export_kwargs = {"format": export_format, "imgsz": imgsz}
            if export_format == "onnx":
                if not CTX.module_available("onnx"):
                    raise RuntimeError("缺少 onnx 依赖，请先安装：pip install -r requirements.txt")
            if export_format == "engine":
                CTX.ensure_tensorrt_env()
                if not torch.cuda.is_available():
                    raise RuntimeError("TensorRT转换需要CUDA环境")
                if not CTX.module_available("tensorrt"):
                    raise RuntimeError("缺少 tensorrt 依赖，无法进行TensorRT转换（请安装 NVIDIA TensorRT Python 包）")
                export_kwargs["device"] = 0
                if precision == "fp16":
                    export_kwargs["half"] = True
                elif precision == "int8":
                    export_kwargs["int8"] = True

            with conversion_status_lock:
                conversion_status = _get_conversion_status_ref(username, create=True)
                conversion_status["progress"] = 55
            exported_path = _export_model_for_conversion(
                model_path,
                export_kwargs,
                temp_export_dirs=temp_export_dirs,
            )
            with conversion_status_lock:
                conversion_status = _get_conversion_status_ref(username, create=True)
                conversion_status["progress"] = 85

            desired_path = _resolve_conversion_output_path(output_path, model_path, export_format)
            os.makedirs(str(desired_path.parent), exist_ok=True)
            if desired_path.resolve() != exported_path.resolve():
                shutil.copy2(str(exported_path), str(desired_path))
                final_path = desired_path.resolve()
            else:
                final_path = exported_path

        ended_at = datetime.datetime.now().isoformat(timespec="seconds")
        duration_ms = int(round((time.perf_counter() - t_record0) * 1000))
        with conversion_status_lock:
            conversion_status = _get_conversion_status_ref(username, create=True)
            conversion_status["status"] = "completed"
            conversion_status["progress"] = 100
            conversion_status["message"] = "转换完成！"
            conversion_status["output_path"] = str(final_path)
            conversion_status["duration_ms"] = duration_ms
            conversion_status["ended_at"] = ended_at

        patch = {
            "status": "completed",
            "message": "转换完成！",
            "output_path": str(final_path),
            "ended_at": ended_at,
            "duration_ms": duration_ms,
        }
        if record_id:
            _update_history_record("conversion", record_id, patch)
    except Exception as e:
        duration_ms = int(round((time.perf_counter() - t_record0) * 1000)) if "t_record0" in locals() else None
        ended_at = datetime.datetime.now().isoformat(timespec="seconds")
        err_msg = f"转换出错: {str(e)}"
        with conversion_status_lock:
            conversion_status = _get_conversion_status_ref(username, create=True)
            conversion_status["status"] = "error"
            conversion_status["message"] = err_msg
            conversion_status["duration_ms"] = duration_ms
            conversion_status["ended_at"] = ended_at
            record_id = conversion_status.get("record_id")
        patch = {"status": "error", "message": err_msg, "ended_at": ended_at, "duration_ms": duration_ms}
        if record_id:
            _update_history_record("conversion", record_id, patch)
    finally:
        for tmp_dir in reversed(temp_export_dirs):
            try:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            except Exception:
                pass
        try:
            CTX.clear_thread_runtime()
        except Exception:
            pass


@bp.route("/api/train/history", methods=["GET"])
def get_train_history():
    _ensure_history_loaded()
    items = _list_owned_history("training")
    return jsonify(list(reversed(items)))


@bp.route("/api/train/history/delete", methods=["POST"])
def delete_train_history():
    _ensure_history_loaded()
    data = request.json or {}
    record_id = data.get("id")
    delete_dir = bool(data.get("delete_dir", True))
    if not record_id:
        return jsonify({"error": "缺少记录ID"}), 400
    deleted_dir = False
    rec = _pop_owned_history_record("training", record_id)
    if rec is None:
        return jsonify({"error": "记录不存在"}), 404
    if delete_dir:
        output_dir = (rec or {}).get("output_dir") or (rec or {}).get("output_path")
        if output_dir:
            try:
                target = Path(str(output_dir)).resolve()
                root = Path(CTX.CURRENT_DIR).resolve()
                try:
                    target.relative_to(root)
                    under_root = True
                except Exception:
                    under_root = False
                if under_root and target.exists() and target.is_dir():
                    shutil.rmtree(str(target), ignore_errors=True)
                    deleted_dir = True
            except Exception:
                deleted_dir = False
    return jsonify({"success": True, "deleted_dir": deleted_dir})


@bp.route("/api/convert/history", methods=["GET"])
def get_convert_history():
    _ensure_history_loaded()
    items = _list_owned_history("conversion")
    return jsonify(list(reversed(items)))


@bp.route("/api/convert/history/delete", methods=["POST"])
def delete_convert_history():
    _ensure_history_loaded()
    data = request.json or {}
    record_id = data.get("id")
    delete_file = bool(data.get("delete_file", False))
    if not record_id:
        return jsonify({"error": "缺少记录ID"}), 400
    deleted_file = False
    rec = _pop_owned_history_record("conversion", record_id)
    if rec is None:
        return jsonify({"error": "记录不存在"}), 404
    if delete_file:
        out_path = (rec or {}).get("output_path")
        if out_path:
            try:
                target = Path(str(out_path)).resolve()
                root = Path(CTX.CURRENT_DIR).resolve()
                try:
                    target.relative_to(root)
                    under_root = True
                except Exception:
                    under_root = False
                if under_root and target.exists() and target.is_file():
                    target.unlink(missing_ok=True)
                    deleted_file = True
            except Exception:
                deleted_file = False
    return jsonify({"success": True, "deleted_file": deleted_file})


@bp.route("/api/validate/history", methods=["GET"])
def get_validate_history():
    _ensure_history_loaded()
    items = _list_owned_history("validation")
    return jsonify(list(reversed(items)))


@bp.route("/api/validate/history/delete", methods=["POST"])
def delete_validate_history():
    _ensure_history_loaded()
    data = request.json or {}
    record_id = data.get("id")
    delete_dir = bool(data.get("delete_dir", True))
    if not record_id:
        return jsonify({"error": "缺少记录ID"}), 400
    deleted_dir = False
    rec = _pop_owned_history_record("validation", record_id)
    if rec is None:
        return jsonify({"error": "记录不存在"}), 404
    if delete_dir:
        save_dir = (rec or {}).get("save_dir")
        if save_dir:
            try:
                target = Path(str(save_dir)).resolve()
                root = Path(CTX.CURRENT_DIR).resolve()
                try:
                    target.relative_to(root)
                    under_root = True
                except Exception:
                    under_root = False
                if under_root and target.exists() and target.is_dir():
                    shutil.rmtree(str(target), ignore_errors=True)
                    deleted_dir = True
            except Exception:
                deleted_dir = False
    return jsonify({"success": True, "deleted_dir": deleted_dir})


@bp.route("/api/train", methods=["POST"])
def train_model():
    username = _current_username()
    workspace_dir = _current_workspace_dir()
    task_type = _current_task_type()
    data = request.json or {}
    yaml_path = data.get("yaml_path")
    epochs = _to_int(data.get("epochs", 100), 100)
    batch_size = _to_int(data.get("batch_size", 16), 16)
    imgsz = _to_int(data.get("imgsz", 640), 640)
    workers = _to_int(data.get("workers", 4), 4)
    model_name = data.get("model_name", "train")
    pretrained_model = data.get("pretrained_model", "yolo11n.pt")
    device_id = data.get("device_id", data.get("deviceId"))
    learning_rate = data.get("learning_rate", data.get("learningRate"))
    train_split = data.get("train_split", data.get("trainSplit", 80))
    output_path = data.get("output_path", data.get("outputPath"))
    augment = data.get("augment")
    if not isinstance(augment, dict):
        augment = None

    with training_status_lock:
        training_status = _get_training_status_ref(username)
        if training_status and str(training_status.get("status") or "") in {"training", "starting"}:
            return jsonify({"error": "训练正在进行中"}), 400
        training_status = _get_training_status_ref(username, create=True)
        training_status.clear()
        training_status.update(_empty_training_status())
        training_status["owner"] = str(username or "")
        training_status["status"] = "starting"
        training_status["progress"] = 0
        training_status["message"] = "准备训练..."
        training_status["task_type"] = task_type

    t = threading.Thread(
        target=train_model_async,
        args=(yaml_path, epochs, batch_size, imgsz, workers, model_name, pretrained_model, device_id, learning_rate, train_split, output_path, augment, username, workspace_dir, task_type),
        daemon=True,
    )
    t.start()
    return jsonify({"success": True})


@bp.route("/api/train/status", methods=["GET"])
def get_train_status():
    username = _current_username()
    with training_status_lock:
        data = _public_status_payload(_get_training_status_ref(username), _empty_training_status)
    return jsonify(data)


@bp.route("/api/validate", methods=["POST"])
def validate_model():
    username = _current_username()
    workspace_dir = _current_workspace_dir()
    task_type = _current_task_type()
    data = request.json or {}
    model_path = data.get("model_path")
    data_path = data.get("data_path")
    conf = _to_float(data.get("conf", 0.25), 0.25)
    iou = _to_float(data.get("iou", 0.7), 0.7)
    if not model_path:
        return jsonify({"error": "缺少模型路径"}), 400

    with validation_status_lock:
        validation_status = _get_validation_status_ref(username)
        if validation_status and str(validation_status.get("status") or "") in {"validating", "starting"}:
            return jsonify({"error": "评估正在进行中"}), 400
        validation_status = _get_validation_status_ref(username, create=True)
        validation_status.clear()
        validation_status.update(_empty_validation_status())
        validation_status["owner"] = str(username or "")
        validation_status["task_type"] = task_type
        validation_status["status"] = "starting"
        validation_status["progress"] = 0
        validation_status["message"] = "准备评估..."
        validation_status["model_path"] = model_path
        validation_status["data_path"] = data_path

    t = threading.Thread(target=validate_model_async, args=(model_path, data_path, conf, iou, username, workspace_dir, task_type), daemon=True)
    t.start()
    return jsonify({"success": True})


@bp.route("/api/validate/status", methods=["GET"])
def get_validate_status():
    username = _current_username()
    with validation_status_lock:
        data = _public_status_payload(_get_validation_status_ref(username), _empty_validation_status)
    return jsonify(data)


@bp.route("/api/eval-plot/<path:relative_path>", methods=["GET"])
def get_eval_plot(relative_path):
    target_path = (Path(CTX.CURRENT_DIR) / str(relative_path)).resolve()
    current_root = Path(CTX.CURRENT_DIR).resolve()
    try:
        target_path.relative_to(current_root)
    except Exception:
        return jsonify({"error": "无效路径"}), 400
    if target_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
        return jsonify({"error": "不支持的文件类型"}), 400
    if not target_path.exists() or not target_path.is_file():
        return jsonify({"error": "文件不存在"}), 404
    return send_from_directory(str(target_path.parent), target_path.name)


def _project_models_root():
    return (Path(CTX.CURRENT_DIR) / str(CTX.MODELS_FOLDER)).resolve()


def _shared_project_models_root():
    try:
        return Path(CTX.get_shared_task_models_dir(CTX.get_current_task())).resolve()
    except Exception:
        return _project_models_root()


def _iter_project_model_roots(include_shared=True):
    roots = []
    current_root = _project_models_root()
    roots.append(current_root)
    if include_shared:
        shared_root = _shared_project_models_root()
        if str(shared_root).lower() != str(current_root).lower():
            roots.append(shared_root)
    return roots


def _resolve_model_input_path(model_path, allowed_exts=None):
    raw = str(model_path or "").strip()
    if not raw:
        raise ValueError("缺少模型路径")
    candidates = []
    path_obj = Path(raw).expanduser()
    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        candidates.append((Path(CTX.CURRENT_DIR) / path_obj).resolve())
        try:
            for root in _iter_project_model_roots(include_shared=True):
                candidates.append((Path(root) / path_obj).resolve())
        except Exception:
            pass
        try:
            candidates.append(path_obj.resolve())
        except Exception:
            pass
    seen = set()
    normalized_exts = {str(ext).lower().lstrip(".") for ext in (allowed_exts or []) if str(ext).strip()}
    for candidate in candidates:
        try:
            resolved = Path(candidate).resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if not resolved.exists() or not resolved.is_file():
                continue
            if normalized_exts and str(resolved.suffix or "").lower().lstrip(".") not in normalized_exts:
                continue
            return resolved
        except Exception:
            continue
    raise FileNotFoundError(f"模型文件不存在: {raw}")


def _resolve_conversion_output_path(output_path, model_path, export_format):
    fmt = _normalize_export_format(export_format) or "engine"
    ext = ".onnx" if fmt == "onnx" else (".rknn" if fmt == "rknn" else (".bmodel" if fmt == "bmodel" else ".engine"))
    stem = Path(str(model_path or "")).stem or "exported"
    if output_path:
        raw = str(output_path).strip()
        if re.fullmatch(r"^[a-zA-Z]:$", raw):
            raw = ""
        if raw:
            desired_path = _ensure_path_under_current_dir(raw, "output_path")
            if desired_path.suffix == "":
                desired_path = desired_path.with_suffix(ext)
            return desired_path
    return (Path(CTX.CURRENT_DIR) / str(CTX.MODELS_FOLDER) / f"{stem}{ext}").resolve()


def _path_has_non_ascii(path_value):
    try:
        return any(ord(ch) > 127 for ch in str(path_value or ""))
    except Exception:
        return False


def _load_yolo_model_uncached(model_path):
    p = Path(str(model_path)).resolve()
    original_torch_load = torch.load
    if p.suffix.lower() == ".pt":
        def patched_torch_load(f, map_location=None, **kwargs):
            return CTX.torch_load_with_compat(original_torch_load, f, map_location=map_location, **kwargs)
        torch.load = patched_torch_load
    try:
        return YOLO(str(p))
    finally:
        torch.load = original_torch_load


def _export_model_for_conversion(model_path, export_kwargs, temp_export_dirs=None):
    source_path = Path(str(model_path)).resolve()
    if os.name == "nt" and _path_has_non_ascii(source_path):
        run_id = uuid.uuid4().hex[:8]
        tmp_dir = (Path(CTX.CURRENT_DIR) / str(CTX.RESULTS_FOLDER) / "_tmp_exports" / run_id).resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_model_name = f"{_sanitize_model_name_for_filename(source_path.stem)}{source_path.suffix or '.pt'}"
        temp_model_path = (tmp_dir / temp_model_name).resolve()
        shutil.copy2(str(source_path), str(temp_model_path))
        if isinstance(temp_export_dirs, list):
            temp_export_dirs.append(tmp_dir)
        model = _load_yolo_model_uncached(temp_model_path)
        return Path(str(model.export(**export_kwargs))).resolve()
    model = CTX.get_cached_model(str(source_path))
    return Path(str(model.export(**export_kwargs))).resolve()


def _project_model_exts():
    return {".pt", ".onnx", ".engine", ".rknn", ".bmodel"}


def _is_model_artifact(path_obj):
    try:
        return path_obj.is_file() and str(path_obj.suffix or "").lower() in _project_model_exts()
    except Exception:
        return False


def _collect_project_model_entry(path_obj, root_dir):
    path_obj = Path(path_obj).resolve()
    root_dir = Path(root_dir).resolve()
    try:
        rel_path = path_obj.relative_to(root_dir).as_posix()
    except Exception:
        rel_path = path_obj.name

    if path_obj.is_file():
        stat = path_obj.stat()
        return {
            "name": path_obj.name,
            "path": str(path_obj),
            "relative_path": rel_path,
            "kind": "file",
            "primary_model": str(path_obj),
            "primary_name": path_obj.name,
            "file_count": 1,
            "size_bytes": int(getattr(stat, "st_size", 0) or 0),
            "mtime": float(getattr(stat, "st_mtime", 0.0) or 0.0),
            "updated_at": datetime.datetime.fromtimestamp(float(getattr(stat, "st_mtime", 0.0) or 0.0)).isoformat(timespec="seconds"),
        }

    candidates = []
    scan_dirs = [path_obj, path_obj / "weights"]
    for scan_dir in scan_dirs:
        try:
            if not scan_dir.exists() or not scan_dir.is_dir():
                continue
            for child in scan_dir.iterdir():
                if _is_model_artifact(child):
                    candidates.append(child.resolve())
        except Exception:
            continue
    dedup = {}
    for child in candidates:
        dedup[str(child).lower()] = child
    candidates = list(dedup.values())

    preferred = None
    preferred_names = ["weights/best.pt", "best.pt", "weights/last.pt", "last.pt"]
    normalized_candidates = {str(c.relative_to(path_obj)).replace("\\", "/").lower(): c for c in candidates}
    for key in preferred_names:
        if key in normalized_candidates:
            preferred = normalized_candidates[key]
            break
    if preferred is None and candidates:
        candidates.sort(key=lambda p: str(p).lower())
        preferred = candidates[0]

    total_size = 0
    mtime = float(getattr(path_obj.stat(), "st_mtime", 0.0) or 0.0)
    for child in candidates:
        try:
            st = child.stat()
            total_size += int(getattr(st, "st_size", 0) or 0)
            mtime = max(mtime, float(getattr(st, "st_mtime", 0.0) or 0.0))
        except Exception:
            continue

    return {
        "name": path_obj.name,
        "path": str(path_obj),
        "relative_path": rel_path,
        "kind": "dir",
        "primary_model": str(preferred) if preferred else "",
        "primary_name": preferred.name if preferred else "",
        "file_count": len(candidates),
        "size_bytes": int(total_size),
        "mtime": mtime,
        "updated_at": datetime.datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else "",
    }


@bp.route("/api/project-models", methods=["GET"])
def list_project_models():
    items = []
    seen_paths = set()
    current_root = _project_models_root()
    for root in _iter_project_model_roots(include_shared=True):
        if not root.exists() or not root.is_dir():
            continue
        try:
            for child in root.iterdir():
                try:
                    child = child.resolve()
                    key = str(child).lower()
                    if key in seen_paths:
                        continue
                    if child.is_dir() or _is_model_artifact(child):
                        entry = _collect_project_model_entry(child, root)
                        entry["scope"] = "shared" if str(root).lower() != str(current_root).lower() else "current"
                        entry["can_delete"] = str(root).lower() == str(current_root).lower()
                        items.append(entry)
                        seen_paths.add(key)
                except Exception:
                    continue
        except Exception:
            pass
    items.sort(key=lambda x: (-float(x.get("mtime") or 0.0), str(x.get("name") or "").lower()))
    return jsonify(items)


@bp.route("/api/project-models/delete", methods=["POST"])
def delete_project_model():
    data = request.json or {}
    target_raw = str(data.get("path") or "").strip()
    if not target_raw:
        return jsonify({"error": "缺少模型路径"}), 400

    root = _project_models_root()
    target = Path(target_raw).expanduser()
    if not target.is_absolute():
        target = (Path(CTX.CURRENT_DIR) / target).resolve()
    else:
        target = target.resolve()

    try:
        target.relative_to(root)
    except Exception:
        return jsonify({"error": "只能删除 models 目录内的模型"}), 400

    if not target.exists():
        return jsonify({"error": "目标不存在"}), 404

    username = _current_username()
    protected_paths = []
    with training_status_lock:
        training_status = _get_training_status_ref(username)
        if training_status and str(training_status.get("status") or "") in {"training", "starting"}:
            current_output = str(training_status.get("output_path") or "").strip()
            if current_output:
                protected_paths.append((current_output, "当前训练输出目录正在使用，暂不允许删除"))
        if CTX.require_admin():
            for key, training_status in training_status_by_user.items():
                if key == _status_owner_key(username):
                    continue
                if str(training_status.get("status") or "") in {"training", "starting"}:
                    current_output = str(training_status.get("output_path") or "").strip()
                    if current_output:
                        protected_paths.append((current_output, "当前训练输出目录正在使用，暂不允许删除"))
    with conversion_status_lock:
        conversion_status = _get_conversion_status_ref(username)
        if conversion_status and str(conversion_status.get("status") or "") in {"converting", "starting"}:
            current_output = str(conversion_status.get("output_path") or "").strip() or _predict_conversion_output_path(conversion_status)
            if current_output:
                protected_paths.append((current_output, "当前转换输出文件正在使用，暂不允许删除"))
        if CTX.require_admin():
            for key, conversion_status in conversion_status_by_user.items():
                if key == _status_owner_key(username):
                    continue
                if str(conversion_status.get("status") or "") in {"converting", "starting"}:
                    current_output = str(conversion_status.get("output_path") or "").strip() or _predict_conversion_output_path(conversion_status)
                    if current_output:
                        protected_paths.append((current_output, "当前转换输出文件正在使用，暂不允许删除"))
    for current_output, message in protected_paths:
        if _paths_overlap(target, current_output):
            return jsonify({"error": message}), 400

    removed_kind = "dir" if target.is_dir() else "file"
    try:
        if target.is_dir():
            shutil.rmtree(str(target), ignore_errors=False)
        else:
            target.unlink()
    except Exception as e:
        return jsonify({"error": f"删除失败: {e}"}), 500

    return jsonify({"success": True, "path": str(target), "kind": removed_kind})


@bp.route("/api/models", methods=["GET"])
def list_models():
    models = []
    seen = set()
    for root in _iter_project_model_roots(include_shared=True):
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            try:
                if p.is_file():
                    key = str(p.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    models.append(str(p.resolve()))
            except Exception:
                continue
    models.sort(key=lambda p: os.path.basename(str(p)).lower())
    return jsonify(models)


@bp.route("/api/pretrained/models", methods=["GET"])
def list_pretrained_models():
    grouped = {"yolov11": [], "yolov26": []}
    seen = {"yolov11": set(), "yolov26": set()}
    for root in _iter_project_model_roots(include_shared=True):
        if not root.exists() or not root.is_dir():
            continue
        try:
            for entry in root.rglob("*.pt"):
                try:
                    if not entry.is_file():
                        continue
                except Exception:
                    continue
                name = entry.name.lower()
                path = str(entry.resolve())
                if "yolo11" in name or "yolov11" in name:
                    if path.lower() not in seen["yolov11"]:
                        grouped["yolov11"].append(path)
                        seen["yolov11"].add(path.lower())
                elif "yolo26" in name or "yolov26" in name:
                    if path.lower() not in seen["yolov26"]:
                        grouped["yolov26"].append(path)
                        seen["yolov26"].add(path.lower())
        except Exception:
            pass
    for k in grouped:
        grouped[k].sort(key=lambda p: os.path.basename(str(p)).lower())
    return jsonify(grouped)


@bp.route("/api/test/models", methods=["GET"])
def list_test_models():
    models = []
    exts = {".pt", ".onnx", ".engine"}
    seen = set()
    for root in _iter_project_model_roots(include_shared=True):
        if not root.exists():
            continue
        try:
            for entry in root.rglob("*"):
                try:
                    if entry.is_file() and entry.suffix.lower() in exts:
                        key = str(entry.resolve()).lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        models.append(str(entry.resolve()))
                except Exception:
                    continue
        except Exception:
            pass
    models.sort(key=lambda p: os.path.basename(str(p)).lower())
    return jsonify(models)


@bp.route("/api/rknn/dataset/generate", methods=["POST"])
def generate_rknn_dataset():
    data = request.json or {}
    yaml_path = data.get("yaml_path") or data.get("dataset_yaml") or data.get("dataset")
    split = data.get("split", "train")
    if not yaml_path:
        return jsonify({"error": "缺少数据集 yaml 路径"}), 400
    try:
        out_path, count = _generate_rknn_dataset_txt(yaml_path, split=split)
        return jsonify({"success": True, "dataset_path": str(out_path), "count": int(count)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/bmodel/calibration/generate", methods=["POST"])
def generate_bmodel_calibration_dataset():
    data = request.json or {}
    yaml_path = data.get("yaml_path") or data.get("dataset_yaml") or data.get("dataset")
    split = data.get("split", "train")
    sample_count = _normalize_bmodel_calibration_count(data.get("sample_count", 100), 100)
    if not yaml_path:
        return jsonify({"error": "缺少数据集 yaml 路径"}), 400
    try:
        out_dir, count = _prepare_bmodel_calibration_dir_from_yaml(yaml_path, split=split, sample_count=sample_count)
        return jsonify({"success": True, "dataset_path": str(out_dir), "count": int(count)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/convert", methods=["POST"])
def convert_model():
    username = _current_username()
    workspace_dir = _current_workspace_dir()
    task_type = _current_task_type()
    data = request.json or {}
    model_path = data.get("model_path")
    export_format = _normalize_export_format(data.get("format"))
    precision = _normalize_precision(data.get("precision", "fp32"))
    output_path = data.get("output_path")
    imgsz = _to_int(data.get("imgsz", 640), 640)
    target_platform = data.get("target_platform", data.get("platform"))
    dataset_path = data.get("dataset_path", data.get("dataset"))
    calibration_table = data.get("calibration_table", data.get("cali_table"))
    calibration_dataset = data.get("calibration_dataset", data.get("calibration_dir"))
    calibration_count = _normalize_bmodel_calibration_count(data.get("calibration_count", 100), 100)

    if not model_path:
        return jsonify({"error": "缺少模型路径"}), 400
    try:
        allowed_exts = ["pt"]
        if export_format in ["rknn", "bmodel"]:
            allowed_exts = ["pt", "onnx"]
        model_path = str(_resolve_model_input_path(model_path, allowed_exts=allowed_exts))
    except Exception as e:
        return jsonify({"error": f"模型文件无效: {str(e)}"}), 400
    if export_format is None:
        return jsonify({"error": "不支持的转换格式"}), 400
    if export_format == "onnx" and precision == "int8":
        return jsonify({"error": "ONNX暂不支持INT8导出"}), 400
    if export_format == "engine":
        CTX.ensure_tensorrt_env()
        if not torch.cuda.is_available():
            return jsonify({"error": "TensorRT转换需要CUDA环境"}), 400
        if not CTX.module_available("tensorrt"):
            return jsonify({"error": "缺少 tensorrt 依赖，无法进行TensorRT转换（请安装 NVIDIA TensorRT Python 包）"}), 400
    if export_format == "rknn":
        if not (CTX.module_available("rknn") or _remote_rknn_enabled()):
            return jsonify({"error": "当前环境无法导出RKNN：需要本机 rknn-toolkit2，或配置远程SSH转换（环境变量 YOLOV11_RKNN_SSH_HOST/USER 或UI「模型转换」页）"}), 400
    if export_format == "bmodel":
        if not (_bmodel_tools_available() or _remote_bmodel_enabled()):
            return jsonify(
                {
                    "error": "当前环境无法导出BModel：请安装本机 tpu-mlir，或配置远程SSH转换（YOLOV11_BMODEL_SSH_HOST/USER 或UI「模型转换」页）"
                }
            ), 400
        if precision == "int8" and not calibration_table and not calibration_dataset:
            return jsonify({"error": "BModel INT8 量化需要提供 calibration_table，或选择标定数据目录"}), 400

    _ensure_history_loaded()
    record_id = uuid.uuid4().hex
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    record = {
        "id": record_id,
        "owner": str(username or ""),
        "task_type": task_type,
        "started_at": started_at,
        "ended_at": None,
        "duration_ms": None,
        "model_path": model_path,
        "format": export_format,
        "precision": precision,
        "imgsz": imgsz,
        "target_platform": str(target_platform or ""),
        "dataset_path": str(dataset_path or ""),
        "calibration_table": str(calibration_table or ""),
        "calibration_dataset": str(calibration_dataset or ""),
        "calibration_count": int(calibration_count),
        "output_path_requested": output_path or "",
        "output_path": None,
        "status": "converting",
        "message": "开始转换...",
    }
    _append_history_record("conversion", record)

    with conversion_status_lock:
        conversion_status = _get_conversion_status_ref(username)
        if conversion_status and str(conversion_status.get("status") or "") in {"converting", "starting"}:
            return jsonify({"error": "模型转换正在进行中"}), 400
        conversion_status = _get_conversion_status_ref(username, create=True)
        conversion_status.clear()
        conversion_status.update(
            {
                "status": "starting",
                "progress": 0,
                "message": "准备转换...",
                "record_id": record_id,
                "owner": username,
                "task_type": task_type,
                "model_path": model_path,
                "format": export_format,
                "precision": precision,
            }
        )

    t = threading.Thread(
        target=convert_model_async,
        args=(
            model_path,
            export_format,
            precision,
            output_path,
            imgsz,
            target_platform,
            dataset_path,
            calibration_table,
            calibration_dataset,
            calibration_count,
            username,
            workspace_dir,
            task_type,
        ),
        daemon=True,
    )
    t.start()
    return jsonify({"success": True})


@bp.route("/api/convert/status", methods=["GET"])
def get_convert_status():
    username = _current_username()
    with conversion_status_lock:
        data = _public_status_payload(_get_conversion_status_ref(username), _empty_conversion_status)
    can_download = False
    output_path = data.get("output_path")
    try:
        if output_path:
            target = Path(str(output_path)).resolve()
            root = Path(CTX.CURRENT_DIR).resolve()
            try:
                target.relative_to(root)
                under_root = True
            except Exception:
                under_root = False
            can_download = under_root and target.exists() and target.is_file()
    except Exception:
        can_download = False
    data["can_download"] = can_download
    if can_download:
        data["download_url"] = "/api/convert/download"
    return jsonify(data)


@bp.route("/api/convert/download", methods=["GET"])
def download_convert_output():
    username = _current_username()
    with conversion_status_lock:
        conversion_status = _get_conversion_status_ref(username)
        status = (conversion_status or {}).get("status")
        output_path = (conversion_status or {}).get("output_path")
    if status != "completed":
        return jsonify({"error": "当前没有可下载的转换结果"}), 400
    if not output_path:
        return jsonify({"error": "输出文件路径缺失"}), 400
    try:
        target = Path(str(output_path)).resolve()
        root = Path(CTX.CURRENT_DIR).resolve()
        try:
            target.relative_to(root)
        except Exception:
            return jsonify({"error": "输出文件不在项目目录内，无法在线下载"}), 400
        if not target.exists() or not target.is_file():
            return jsonify({"error": "输出文件不存在"}), 404
        return send_from_directory(str(target.parent), target.name, as_attachment=True, download_name=target.name)
    except Exception as e:
        return jsonify({"error": f"下载失败: {str(e)}"}), 500


@bp.route("/api/settings/bmodel-check", methods=["POST"])
def check_bmodel_remote_tools():
    if not CTX.is_authenticated():
        return jsonify({"error": "未登录"}), 401
    if not CTX.require_admin():
        return jsonify({"error": "无权限"}), 403
    data = request.json or {}
    override = data.get("bmodel_ssh") if isinstance(data, dict) else None
    cfg = _bmodel_ssh_with_override(override)
    remote_python = str(cfg.get("python") or "python3").strip() or "python3"
    remote_transform = str(cfg.get("transform") or "model_transform.py").strip() or "model_transform.py"
    remote_deploy = str(cfg.get("deploy") or "model_deploy.py").strip() or "model_deploy.py"
    remote_calibration = str(cfg.get("calibration") or "run_calibration.py").strip() or "run_calibration.py"
    checks = {
        "ssh": {"ok": False, "message": ""},
        "python": {"ok": False, "message": ""},
        "transform": {"ok": False, "message": ""},
        "deploy": {"ok": False, "message": ""},
        "calibration": {"ok": False, "message": ""},
        "onnxsim": {"ok": False, "message": ""},
    }
    suggestions = []

    def _build_suggestions():
        host = str(cfg.get("host") or "")
        user = str(cfg.get("user") or "")
        port = str(cfg.get("port") or "").strip()
        key = str(cfg.get("key") or "").strip()
        target = f"{user}@{host}" if user and host else "<user>@<host>"
        ssh_prefix = "ssh"
        if port:
            ssh_prefix += f" -p {shlex.quote(port)}"
        if key:
            ssh_prefix += f" -i {shlex.quote(key)}"
        remote_python_q = shlex.quote(remote_python)
        transform_q = shlex.quote(remote_transform)
        deploy_q = shlex.quote(remote_deploy)
        calibration_q = shlex.quote(remote_calibration)
        out = []
        if not checks["ssh"]["ok"]:
            out.append(f"本机先验证连通: {ssh_prefix} {target} 'echo ok'")
        if checks["ssh"]["ok"] and not checks["python"]["ok"]:
            out.append(f"远端查看Python: {ssh_prefix} {target} 'command -v {remote_python_q} || which python3 || python3 -V'")
        if checks["ssh"]["ok"] and checks["python"]["ok"] and not checks["transform"]["ok"]:
            out.append(f"远端检查transform: {ssh_prefix} {target} 'command -v {transform_q} || ls -l {transform_q}'")
        if checks["ssh"]["ok"] and checks["python"]["ok"] and not checks["deploy"]["ok"]:
            out.append(f"远端检查deploy: {ssh_prefix} {target} 'command -v {deploy_q} || ls -l {deploy_q}'")
        if checks["ssh"]["ok"] and checks["python"]["ok"] and not checks["calibration"]["ok"]:
            out.append(f"远端检查calibration: {ssh_prefix} {target} 'command -v {calibration_q} || ls -l {calibration_q}'")
        if checks["ssh"]["ok"] and checks["python"]["ok"] and not checks["onnxsim"]["ok"]:
            out.append(f"远端安装 onnxsim: {ssh_prefix} {target} '{remote_python_q} -m pip install onnxsim'")
        if checks["ssh"]["ok"] and (not checks["transform"]["ok"] or not checks["deploy"]["ok"] or not checks["calibration"]["ok"]):
            out.append(
                "若命令不在PATH，可在远端加入PATH后再测: "
                f"{ssh_prefix} {target} 'export PATH=$PATH:/path/to/tpu-mlir/bin; command -v {transform_q}; command -v {deploy_q}; command -v {calibration_q}'"
            )
        return out

    def fail(msg, code=400):
        nonlocal suggestions
        suggestions = _build_suggestions()
        return (
            jsonify(
                {
                    "ok": False,
                    "error": msg,
                    "checks": checks,
                    "suggestions": suggestions,
                    "host": str(cfg.get("host") or ""),
                    "user": str(cfg.get("user") or ""),
                }
            ),
            code,
        )

    try:
        _build_ssh_target(cfg)
    except Exception as e:
        checks["ssh"]["message"] = f"参数无效: {str(e)}"
        return fail(f"BModel SSH 参数无效: {str(e)}", 400)

    try:
        _remote_exec(cfg, "echo __BMODEL_SSH_OK__", timeout=20)
    except Exception as e:
        checks["ssh"]["message"] = f"连接失败: {str(e)}"
        return fail(f"SSH连接失败: {str(e)}", 400)
    checks["ssh"]["ok"] = True
    checks["ssh"]["message"] = "连接成功"

    py_check_cmd = f"command -v {shlex.quote(remote_python)} >/dev/null 2>&1 && echo OK || echo FAIL"
    py_check = _remote_exec(cfg, py_check_cmd, timeout=20).strip()
    if py_check != "OK":
        checks["python"]["message"] = f"未找到解释器: {remote_python}"
        return fail(f"远端未找到Python解释器: {remote_python}", 400)
    checks["python"]["ok"] = True
    checks["python"]["message"] = f"可用: {remote_python}"

    def _tool_check_cmd(tool):
        t = str(tool or "").strip()
        if not t:
            return "echo FAIL"
        # Explicit paths are checked as files; bare names are checked in PATH.
        if "/" in t or "\\" in t:
            return f"[ -f {shlex.quote(t)} ] && echo OK || echo FAIL"
        return f"command -v {shlex.quote(t)} >/dev/null 2>&1 && echo OK || echo FAIL"

    transform_ok = _remote_exec(cfg, _tool_check_cmd(remote_transform), timeout=20).strip() == "OK"
    deploy_ok = _remote_exec(cfg, _tool_check_cmd(remote_deploy), timeout=20).strip() == "OK"
    calibration_ok = _remote_exec(cfg, _tool_check_cmd(remote_calibration), timeout=20).strip() == "OK"
    checks["transform"]["ok"] = bool(transform_ok)
    checks["transform"]["message"] = f"{'可用' if transform_ok else '不可用'}: {remote_transform}"
    checks["deploy"]["ok"] = bool(deploy_ok)
    checks["deploy"]["message"] = f"{'可用' if deploy_ok else '不可用'}: {remote_deploy}"
    checks["calibration"]["ok"] = bool(calibration_ok)
    checks["calibration"]["message"] = f"{'可用' if calibration_ok else '不可用'}: {remote_calibration}"
    if not transform_ok or not deploy_ok:
        missing = []
        if not transform_ok:
            missing.append(f"transform={remote_transform}")
        if not deploy_ok:
            missing.append(f"deploy={remote_deploy}")
        return fail("远端工具不可用: " + ", ".join(missing), 400)

    onnxsim_probe = 'import onnxsim; print("OK")'
    onnxsim_cmd = f"{shlex.quote(remote_python)} -c {shlex.quote(onnxsim_probe)}"
    try:
        onnxsim_ok = _remote_exec(cfg, onnxsim_cmd, timeout=20).strip() == "OK"
    except Exception as e:
        checks["onnxsim"]["message"] = f"不可用: {str(e)}"
        return fail("远端 Python 环境缺少 onnxsim，请先安装：onnxsim", 400)
    checks["onnxsim"]["ok"] = bool(onnxsim_ok)
    checks["onnxsim"]["message"] = "可导入: onnxsim"
    if not onnxsim_ok:
        return fail("远端 Python 环境缺少 onnxsim，请先安装：onnxsim", 400)

    return jsonify(
        {
            "ok": True,
            "message": "连接成功，远端工具与依赖可用" if calibration_ok else "连接成功，基础转换工具可用；自动生成校准表工具不可用",
            "python": remote_python,
            "transform": remote_transform,
            "deploy": remote_deploy,
            "calibration": remote_calibration,
            "host": str(cfg.get("host") or ""),
            "user": str(cfg.get("user") or ""),
            "checks": checks,
            "suggestions": [] if calibration_ok else _build_suggestions(),
        }
    )


@bp.route("/api/settings/rknn-check", methods=["POST"])
def check_rknn_remote_tools():
    if not CTX.is_authenticated():
        return jsonify({"error": "未登录"}), 401
    if not CTX.require_admin():
        return jsonify({"error": "无权限"}), 403
    data = request.json or {}
    override = data.get("rknn_ssh") if isinstance(data, dict) else None
    cfg = _rknn_ssh_with_override(override)
    checks = {
        "ssh": {"ok": False, "message": ""},
        "python": {"ok": False, "message": ""},
        "rknn": {"ok": False, "message": ""},
        "onnx": {"ok": False, "message": ""},
    }
    suggestions = []

    remote_python = str(cfg.get("python") or "python3").strip() or "python3"

    def _build_suggestions():
        host = str(cfg.get("host") or "")
        user = str(cfg.get("user") or "")
        port = str(cfg.get("port") or "").strip()
        key = str(cfg.get("key") or "").strip()
        target = f"{user}@{host}" if user and host else "<user>@<host>"
        ssh_prefix = "ssh"
        if port:
            ssh_prefix += f" -p {shlex.quote(port)}"
        if key:
            ssh_prefix += f" -i {shlex.quote(key)}"
        py_q = shlex.quote(remote_python)
        out = []
        if not checks["ssh"]["ok"]:
            out.append(f"本机先验证连通: {ssh_prefix} {target} 'echo ok'")
        if checks["ssh"]["ok"] and not checks["python"]["ok"]:
            out.append(f"远端查看Python: {ssh_prefix} {target} 'command -v {py_q} || which python3 || python3 -V'")
        if checks["ssh"]["ok"] and checks["python"]["ok"] and not checks["rknn"]["ok"]:
            out.append(
                f"远端验证RKNN: {ssh_prefix} {target} "
                f"\"{remote_python} -c 'from rknn.api import RKNN; print(\\\"ok\\\")'\""
            )
            out.append(
                "若模块缺失，请在远端安装并激活环境后重测: "
                f"{ssh_prefix} {target} '{remote_python} -m pip show rknn-toolkit2 || true'"
            )
        if checks["ssh"]["ok"] and checks["python"]["ok"] and checks["rknn"]["ok"] and not checks["onnx"]["ok"]:
            out.append(
                "远端 onnx 版本需降级到 1.15.x: "
                f"{ssh_prefix} {target} '{remote_python} -m pip install \"onnx==1.15.0\"'"
            )
        return out

    def fail(msg, code=400):
        nonlocal suggestions
        suggestions = _build_suggestions()
        return (
            jsonify(
                {
                    "ok": False,
                    "error": msg,
                    "checks": checks,
                    "suggestions": suggestions,
                    "host": str(cfg.get("host") or ""),
                    "user": str(cfg.get("user") or ""),
                }
            ),
            code,
        )

    try:
        _build_ssh_target(cfg)
    except Exception as e:
        checks["ssh"]["message"] = f"参数无效: {str(e)}"
        return fail(f"RKNN SSH 参数无效: {str(e)}", 400)

    try:
        _remote_exec(cfg, "echo __RKNN_SSH_OK__", timeout=20)
    except Exception as e:
        checks["ssh"]["message"] = f"连接失败: {str(e)}"
        return fail(f"SSH连接失败: {str(e)}", 400)
    checks["ssh"]["ok"] = True
    checks["ssh"]["message"] = "连接成功"

    py_check_cmd = f"command -v {shlex.quote(remote_python)} >/dev/null 2>&1 && echo OK || echo FAIL"
    py_check = _remote_exec(cfg, py_check_cmd, timeout=20).strip()
    if py_check != "OK":
        checks["python"]["message"] = f"未找到解释器: {remote_python}"
        return fail(f"远端未找到Python解释器: {remote_python}", 400)
    checks["python"]["ok"] = True
    checks["python"]["message"] = f"可用: {remote_python}"

    rknn_check_cmd = (
        f"{shlex.quote(remote_python)} -c "
        "\"from rknn.api import RKNN; print('OK')\" >/dev/null 2>&1 && echo OK || echo FAIL"
    )
    rknn_ok = _remote_exec(cfg, rknn_check_cmd, timeout=30).strip() == "OK"
    checks["rknn"]["ok"] = bool(rknn_ok)
    checks["rknn"]["message"] = "可导入 rknn.api" if rknn_ok else "无法导入 rknn.api"
    if not rknn_ok:
        return fail("远端环境缺少可用的 rknn-toolkit2（或Python环境未激活）", 400)

    onnx_check_cmd = (
        f"{shlex.quote(remote_python)} -c "
        "\"import onnx; v=str(getattr(onnx,'__version__','')); "
        "print(('OK ' if hasattr(onnx,'mapping') else 'FAIL ') + v)\""
    )
    onnx_line = _remote_exec(cfg, onnx_check_cmd, timeout=30).strip().splitlines()[-1].strip()
    onnx_ok = onnx_line.startswith("OK ")
    onnx_version = onnx_line[3:].strip() if len(onnx_line) > 3 else ""
    checks["onnx"]["ok"] = bool(onnx_ok)
    checks["onnx"]["message"] = (
        f"兼容: onnx=={onnx_version or 'unknown'}"
        if onnx_ok
        else _rknn_onnx_compat_message(onnx_version or "unknown")
    )
    if not onnx_ok:
        return fail("远端 onnx 与 rknn-toolkit2 不兼容", 400)

    return jsonify(
        {
            "ok": True,
            "message": "连接成功，远端RKNN环境可用",
            "python": remote_python,
            "host": str(cfg.get("host") or ""),
            "user": str(cfg.get("user") or ""),
            "checks": checks,
            "suggestions": [],
        }
    )
