from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_CONFIG_KEYS = {
    "create_bias",
    "single_doc_create_bias",
    "stale_create_bias",
    "stale_single_doc_create_bias",
    "stale_age_threshold",
    "ambiguous_create_bias",
    "ambiguity_threshold",
}

RETRIEVAL_CONFIG_KEYS = {
    "mode",
    "anchor_title_bonus",
    "anchor_context_bonus",
    "anchor_token_bonus",
}

SKIP_KEYS = {
    "artifact_path",
    "base_top_k",
    "base_config",
    "base_dev",
    "base_dev_stats",
    "base_test",
    "base_test_stats",
    "best",
    "best_selection_source",
    "device",
    "benchmark_dir",
    "checkpoint_path",
    "dev",
    "dev_stats",
    "improvement",
    "local_fragment_bridge",
    "model",
    "model_type",
    "results",
    "output_path",
    "top_k",
    "prediction_dir",
    "qwen_benchmark_dir",
    "reference_dev",
    "reference_test",
    "require_candidate_change_for_override",
    "reretrieve_link_margin_threshold",
    "reretrieve_min_age",
    "reretrieve_state_update",
    "reretrieve_top_k",
    "results",
    "retrieval_config",
    "router_variant",
    "train_stats",
    "trainable",
    "trainable_reason",
    "test",
    "test_stats",
}


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"override must look like key=value, got: {raw}")
    key, raw_value = raw.split("=", maxsplit=1)
    key = key.strip()
    raw_value = raw_value.strip()
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def _set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: dict[str, Any] = payload
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError(f"invalid override key: {dotted_key}")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_stringify(item) for item in value)
    return str(value)


def _add_value_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, str) and not value:
        return
    command.extend([flag, _stringify(value)])


def _build_command(payload: dict[str, Any], *, output_path: str, device: str) -> list[str]:
    base_config = dict(payload.get("base_config", {}))
    retrieval_config = dict(payload.get("retrieval_config", {}))

    command = [
        sys.executable,
        "-m",
        "mcdecr.scripts.new_task.evaluate_hierarchical_persistent_memory",
        "--checkpoint-path",
        str(payload["checkpoint_path"]),
        "--benchmark-dir",
        str(payload["benchmark_dir"]),
        "--device",
        device,
        "--output-path",
        output_path,
    ]

    _add_value_arg(command, "--qwen-benchmark-dir", payload.get("qwen_benchmark_dir"))

    for key in (
        "base_top_k",
        "reretrieve_top_k",
        "reretrieve_min_age",
        "reretrieve_link_margin_threshold",
        "reretrieve_state_update",
        "require_candidate_change_for_override",
    ):
        _add_value_arg(command, f"--{key.replace('_', '-')}", payload.get(key))

    for key in (
        "create_bias",
        "single_doc_create_bias",
        "stale_create_bias",
        "stale_single_doc_create_bias",
        "stale_age_threshold",
        "ambiguous_create_bias",
        "ambiguity_threshold",
    ):
        _add_value_arg(command, f"--{key.replace('_', '-')}", base_config.get(key))

    retrieval_mode = retrieval_config.get("mode", "base")
    _add_value_arg(command, "--retrieval-mode", retrieval_mode)
    if retrieval_mode != "base":
        for key in ("anchor_title_bonus", "anchor_context_bonus", "anchor_token_bonus"):
            _add_value_arg(command, f"--{key.replace('_', '-')}", retrieval_config.get(key))

    global_consensus_model_path = payload.get("global_consensus_model_path", "")
    if global_consensus_model_path:
        _add_value_arg(
            command,
            "--global-consensus-thresholds",
            payload.get("global_consensus_thresholds", payload.get("global_consensus_threshold")),
        )
        _add_value_arg(
            command,
            "--global-relink-consensus-thresholds",
            payload.get("global_relink_consensus_thresholds", payload.get("global_relink_consensus_threshold")),
        )
    else:
        _add_value_arg(command, "--global-score-thresholds", [10**9])
        _add_value_arg(command, "--global-margin-thresholds", [10**9])
        _add_value_arg(command, "--global-require-title-match-options", [False])
        _add_value_arg(command, "--global-min-token-overlaps", [10**9])

    if payload.get("fragment_bridge_mode"):
        command.append("--local-fragment-bridge")

    for key, value in payload.items():
        if key in SKIP_KEYS:
            continue
        if key in BASE_CONFIG_KEYS or key in RETRIEVAL_CONFIG_KEYS:
            continue
        _add_value_arg(command, f"--{key.replace('_', '-')}", value)

    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--effective-config-path", default="")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    for override_raw in args.set:
        key, value = _parse_override(override_raw)
        _set_nested_value(payload, key, value)

    payload["output_path"] = args.output_path

    if args.effective_config_path:
        effective_config_path = Path(args.effective_config_path)
        if not effective_config_path.is_absolute():
            effective_config_path = REPO_ROOT / effective_config_path
        effective_config_path.parent.mkdir(parents=True, exist_ok=True)
        effective_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    command = _build_command(payload, output_path=args.output_path, device=args.device)
    if args.print_command or args.dry_run:
        print(" ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
