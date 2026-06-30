from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from new_task.streaming_memory import (
    EpisodePrediction,
    MemorySlot,
    StreamingMentionRecord,
    anchor_overlap_jaccard,
    evaluate_episode_predictions,
    load_records,
    memory_candidate_feature_group_indices,
    memory_candidate_feature_dim,
    memory_candidate_feature_names,
    memory_candidate_features,
    memory_maturity_features,
    new_memory_slot,
    record_anchor_title,
    record_anchor_token_set,
    record_context_titles,
    save_json,
    shortlist_aggregate_feature_dim,
    shortlist_candidate_row_dim,
    update_memory_slot,
    visual_reliability_score,
)


def _group_by_episode(records: list[StreamingMentionRecord]) -> dict[str, list[StreamingMentionRecord]]:
    episodes: dict[str, list[StreamingMentionRecord]] = defaultdict(list)
    for record in records:
        episodes[record.episode_id].append(record)
    for episode_records in episodes.values():
        episode_records.sort(key=lambda item: item.stream_index)
    return episodes


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_slot(memory_id: str, record: StreamingMentionRecord) -> MemorySlot:
    return new_memory_slot(memory_id, record)


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "base"
    anchor_title_bonus: float = 0.0
    anchor_context_bonus: float = 0.0
    anchor_token_bonus: float = 0.0
    disable_maturity_signals: bool = False
    disable_maturity_scoring: bool = False
    disable_visual_signals: bool = False
    disable_anchor_history: bool = False
    visual_update_mode: str = "always"
    visual_reliability_threshold: float = 0.0
    visual_update_rate: float = 1.0


def _update_slot(
    slot: MemorySlot,
    record: StreamingMentionRecord,
    retrieval_config: RetrievalConfig | None = None,
) -> None:
    config = retrieval_config or BASE_RETRIEVAL_CONFIG
    update_memory_slot(
        slot,
        record,
        visual_update_mode=config.visual_update_mode,
        visual_reliability_threshold=config.visual_reliability_threshold,
        visual_update_rate=config.visual_update_rate,
    )


BASE_RETRIEVAL_CONFIG = RetrievalConfig()

STATE_AWARE_AGE_BUCKETS = ("0", "1", "2-3", "4-7", "8+")
STATE_AWARE_DOC_BUCKETS = ("1", "2", "3+")
STATE_AWARE_MEMORY_BUCKETS = ("1", "2-3", "4-7", "8+")
STATE_AWARE_MARGIN_BUCKETS = ("<0.25", "0.25-0.5", "0.5-1.0", "1.0+")


def _bucket_age(value: int | None) -> str:
    if value is None or value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return "8+"


def _bucket_doc_count(value: int | None) -> str:
    if value is None or value <= 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def _bucket_memory_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return "8+"


def _bucket_margin(create_logit: float | None, best_link_logit: float | None) -> str:
    if create_logit is None or best_link_logit is None:
        return "1.0+"
    margin = abs(create_logit - best_link_logit)
    if margin < 0.25:
        return "<0.25"
    if margin < 0.5:
        return "0.25-0.5"
    if margin < 1.0:
        return "0.5-1.0"
    return "1.0+"


def _empty_state_aware_bias_config() -> dict[str, dict[str, float]]:
    return {
        "age_biases": {},
        "doc_biases": {},
        "memory_biases": {},
        "margin_biases": {},
    }


def _normalize_state_aware_bias_config(config: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    normalized = _empty_state_aware_bias_config()
    if not config:
        return normalized

    family_specs = (
        ("age_biases", STATE_AWARE_AGE_BUCKETS),
        ("doc_biases", STATE_AWARE_DOC_BUCKETS),
        ("memory_biases", STATE_AWARE_MEMORY_BUCKETS),
        ("margin_biases", STATE_AWARE_MARGIN_BUCKETS),
    )
    for key, allowed_buckets in family_specs:
        raw_family = config.get(key)
        if raw_family is None and key.endswith("_biases"):
            raw_family = config.get(key.replace("_biases", ""))
        if not isinstance(raw_family, dict):
            continue
        for bucket, value in raw_family.items():
            if bucket not in allowed_buckets:
                continue
            try:
                normalized[key][bucket] = float(value)
            except (TypeError, ValueError):
                continue
    return normalized


def _state_aware_bucket_snapshot(
    num_slots_before: int,
    best_candidate_doc_count_before: int | None,
    best_candidate_age_before: int | None,
    raw_create_logit: float | None,
    best_link_logit: float | None,
) -> dict[str, str]:
    return {
        "age_bucket": _bucket_age(best_candidate_age_before),
        "doc_bucket": _bucket_doc_count(best_candidate_doc_count_before),
        "memory_bucket": _bucket_memory_count(num_slots_before),
        "margin_bucket": _bucket_margin(raw_create_logit, best_link_logit),
    }


def _compose_create_bias(
    *,
    create_bias: float,
    single_doc_create_bias: float,
    stale_create_bias: float,
    stale_age_threshold: int,
    ambiguous_create_bias: float,
    ambiguity_threshold: float,
    num_slots_before: int,
    best_candidate_doc_count_before: int | None,
    best_candidate_age_before: int | None,
    raw_create_logit: float | None,
    best_link_logit: float | None,
    state_aware_bias_config: dict[str, dict[str, float]] | None = None,
) -> tuple[float, dict[str, str]]:
    applied_create_bias = float(create_bias)
    if best_candidate_doc_count_before is not None and best_candidate_doc_count_before <= 1:
        applied_create_bias += float(single_doc_create_bias)
    if (
        best_candidate_age_before is not None
        and stale_age_threshold >= 0
        and best_candidate_age_before >= stale_age_threshold
    ):
        applied_create_bias += float(stale_create_bias)
    if (
        best_link_logit is not None
        and raw_create_logit is not None
        and abs(raw_create_logit - best_link_logit) <= ambiguity_threshold
    ):
        applied_create_bias += float(ambiguous_create_bias)

    bucket_snapshot = _state_aware_bucket_snapshot(
        num_slots_before=num_slots_before,
        best_candidate_doc_count_before=best_candidate_doc_count_before,
        best_candidate_age_before=best_candidate_age_before,
        raw_create_logit=raw_create_logit,
        best_link_logit=best_link_logit,
    )
    normalized_state_config = state_aware_bias_config or _empty_state_aware_bias_config()
    applied_create_bias += normalized_state_config["age_biases"].get(bucket_snapshot["age_bucket"], 0.0)
    applied_create_bias += normalized_state_config["doc_biases"].get(bucket_snapshot["doc_bucket"], 0.0)
    applied_create_bias += normalized_state_config["memory_biases"].get(bucket_snapshot["memory_bucket"], 0.0)
    applied_create_bias += normalized_state_config["margin_biases"].get(bucket_snapshot["margin_bucket"], 0.0)
    return applied_create_bias, bucket_snapshot


def _anchor_retrieval_bonus(
    slot: MemorySlot,
    record: StreamingMentionRecord,
    retrieval_config: RetrievalConfig,
) -> float:
    if retrieval_config.mode != "anchor":
        return 0.0

    record_title = record_anchor_title(record)
    record_context = set(record_context_titles(record))
    record_tokens = record_anchor_token_set(record)
    title_match = 1.0 if record_title and record_title in slot.anchor_titles else 0.0
    context_overlap = anchor_overlap_jaccard(slot.context_titles, record_context)
    token_overlap = anchor_overlap_jaccard(slot.anchor_token_set, record_tokens)
    return (
        retrieval_config.anchor_title_bonus * title_match
        + retrieval_config.anchor_context_bonus * context_overlap
        + retrieval_config.anchor_token_bonus * token_overlap
    )


def _apply_feature_ablation(features: np.ndarray, retrieval_config: RetrievalConfig) -> np.ndarray:
    if not (
        retrieval_config.disable_maturity_signals
        or retrieval_config.disable_maturity_scoring
        or retrieval_config.disable_visual_signals
        or retrieval_config.disable_anchor_history
    ):
        return features

    ablated = features.copy()
    groups = memory_candidate_feature_group_indices()
    indices: set[int] = set()
    if retrieval_config.disable_maturity_signals:
        indices.update(groups["maturity"])
    if retrieval_config.disable_maturity_scoring:
        indices.update(groups["maturity_scoring"])
    if retrieval_config.disable_visual_signals:
        indices.update(groups["visual"])
    if retrieval_config.disable_anchor_history:
        indices.update(groups["anchor_history"])
    for index in indices:
        if 0 <= index < ablated.shape[0]:
            ablated[index] = 0.0
    return ablated


def _retrieval_score(
    slot: MemorySlot,
    record: StreamingMentionRecord,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
) -> float:
    features = _apply_feature_ablation(memory_candidate_features(slot, record), retrieval_config)
    exemplar_text = float(features[2])
    exemplar_visual = float(features[3])
    same_lemma = float(features[6])
    return (
        exemplar_text
        + 0.15 * exemplar_visual
        + 0.03 * same_lemma
        + _anchor_retrieval_bonus(slot, record, retrieval_config)
    )


def _ranked_shortlist(
    slots: list[tuple[str, MemorySlot]],
    record: StreamingMentionRecord,
    top_k: int,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
) -> list[tuple[str, np.ndarray, float]]:
    scored = []
    for memory_id, slot in slots:
        features = _apply_feature_ablation(memory_candidate_features(slot, record), retrieval_config)
        retrieval = _retrieval_score(slot, record, retrieval_config=retrieval_config)
        scored.append((memory_id, features, retrieval))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:top_k]


def _aggregate_shortlist_features(
    shortlist: list[tuple[str, np.ndarray, float]],
    num_slots: int,
) -> np.ndarray:
    if not shortlist:
        return np.zeros(shortlist_aggregate_feature_dim(), dtype=np.float32)
    retrievals = [row[2] for row in shortlist]
    best = retrievals[0]
    second = retrievals[1] if len(retrievals) > 1 else 0.0
    mean_topk = float(np.mean(retrievals))
    best_features = shortlist[0][1]
    second_features = shortlist[1][1] if len(shortlist) > 1 else np.zeros_like(best_features)
    aggregate = np.concatenate(
        [
            best_features,
            np.asarray(
                [
                    best,
                    second,
                    best - second,
                    mean_topk,
                    float(len(shortlist)),
                    float(np.log1p(num_slots)),
                ],
                dtype=np.float32,
            ),
            best_features - second_features,
        ],
        axis=0,
    )
    return aggregate.astype(np.float32)


def _candidate_row(
    slots: list[tuple[str, MemorySlot]],
    shortlist: list[tuple[str, np.ndarray, float]],
    memory_id: str,
    features: np.ndarray,
    retrieval: float,
) -> np.ndarray:
    best_retrieval = shortlist[0][2]
    second_retrieval = shortlist[1][2] if len(shortlist) > 1 else 0.0
    slot_size = float(
        np.log1p(next(len(slot.mentions) for candidate_id, slot in slots if candidate_id == memory_id))
    )
    row = np.concatenate(
        [
            features,
            np.asarray(
                [
                    retrieval,
                    best_retrieval,
                    retrieval - best_retrieval,
                    best_retrieval - second_retrieval,
                    slot_size,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return row.astype(np.float32)


@dataclass
class JointRouterData:
    aggregate_x: np.ndarray
    candidate_x: np.ndarray
    candidate_mask: np.ndarray
    target_y: np.ndarray
    pollution_y: np.ndarray
    fragmentation_y: np.ndarray
    shortlist_sizes: np.ndarray
    num_decisions: int
    num_training_examples: int
    dropped_missing_gold: int


def _build_example_weights(
    aggregate_x: np.ndarray,
    candidate_x: np.ndarray,
    target_y: np.ndarray,
    ambiguity_alpha: float,
    single_doc_alpha: float,
) -> np.ndarray:
    weights = np.ones(target_y.shape[0], dtype=np.float32)
    if weights.size == 0:
        return weights

    retrieval_gap_index = memory_candidate_feature_dim() + 2
    ambiguity_gap = np.maximum(aggregate_x[:, retrieval_gap_index], 0.0)
    if ambiguity_alpha > 0.0:
        weights += ambiguity_alpha * np.exp(-ambiguity_gap)

    top_unique_doc_log = candidate_x[:, 0, 10]
    top_unique_doc_count = np.expm1(np.maximum(top_unique_doc_log, 0.0))
    if single_doc_alpha > 0.0:
        weights += single_doc_alpha * (top_unique_doc_count <= 1.05).astype(np.float32)

    create_mask = (target_y == 0).astype(np.float32)
    if ambiguity_alpha > 0.0 or single_doc_alpha > 0.0:
        weights += 0.25 * create_mask
    return weights


def _build_attachment_guard_weights(
    aggregate_x: np.ndarray,
    candidate_x: np.ndarray,
    target_y: np.ndarray,
    ambiguity_guard_alpha: float,
    single_doc_guard_alpha: float,
) -> np.ndarray:
    guard_weights = np.zeros(target_y.shape[0], dtype=np.float32)
    if guard_weights.size == 0:
        return guard_weights

    retrieval_gap_index = memory_candidate_feature_dim() + 2
    ambiguity_gap = np.maximum(aggregate_x[:, retrieval_gap_index], 0.0)
    if ambiguity_guard_alpha > 0.0:
        guard_weights += ambiguity_guard_alpha * np.exp(-ambiguity_gap)

    top_unique_doc_log = candidate_x[:, 0, 10]
    top_unique_doc_count = np.expm1(np.maximum(top_unique_doc_log, 0.0))
    if single_doc_guard_alpha > 0.0:
        guard_weights += single_doc_guard_alpha * (top_unique_doc_count <= 1.05).astype(np.float32)

    guard_weights *= (target_y == 0).astype(np.float32)
    return guard_weights.astype(np.float32)


def build_joint_training_data(
    records: list[StreamingMentionRecord],
    top_k: int,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    progress_every: int = 0,
) -> JointRouterData:
    episodes = _group_by_episode(records)
    aggregate_rows: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    target_y: list[int] = []
    pollution_targets: list[np.ndarray] = []
    fragmentation_targets: list[float] = []
    shortlist_sizes: list[int] = []
    num_decisions = 0
    dropped_missing_gold = 0

    total_episodes = len(episodes)
    for episode_index, episode_records in enumerate(episodes.values(), start=1):
        if progress_every > 0 and (episode_index == 1 or episode_index % progress_every == 0):
            print(
                f"build_joint_training_data episode={episode_index}/{total_episodes} "
                f"examples={len(target_y)} decisions={num_decisions}",
                flush=True,
            )
        gold_slots: dict[str, MemorySlot] = {}
        for record in episode_records:
            shortlist = _ranked_shortlist(
                list(gold_slots.items()),
                record,
                top_k,
                retrieval_config=retrieval_config,
            )
            if shortlist:
                num_decisions += 1
                candidate_matrix = np.zeros((top_k, shortlist_candidate_row_dim()), dtype=np.float32)
                candidate_mask = np.zeros(top_k, dtype=bool)
                gold_position = -1
                slots_snapshot = list(gold_slots.items())
                for idx, (memory_id, features, retrieval) in enumerate(shortlist):
                    candidate_matrix[idx] = _candidate_row(
                        slots_snapshot,
                        shortlist,
                        memory_id,
                        features,
                        retrieval,
                    )
                    candidate_mask[idx] = True
                    if memory_id == record.gold_cluster:
                        gold_position = idx

                if record.gold_action == "LINK" and gold_position < 0:
                    dropped_missing_gold += 1
                else:
                    target = 0 if record.gold_action == "CREATE" else gold_position + 1
                    pollution_target = np.ones(top_k, dtype=np.float32)
                    if target > 0:
                        pollution_target[target - 1] = 0.0
                    aggregate_rows.append(_aggregate_shortlist_features(shortlist, num_slots=len(gold_slots)))
                    candidate_rows.append(candidate_matrix)
                    candidate_masks.append(candidate_mask)
                    target_y.append(target)
                    pollution_targets.append(pollution_target)
                    fragmentation_targets.append(float(record.gold_action == "LINK"))
                    shortlist_sizes.append(len(shortlist))

            if record.gold_cluster in gold_slots:
                _update_slot(gold_slots[record.gold_cluster], record)
            else:
                gold_slots[record.gold_cluster] = _new_slot(record.gold_cluster, record)

    return JointRouterData(
        aggregate_x=np.stack(aggregate_rows, axis=0),
        candidate_x=np.stack(candidate_rows, axis=0),
        candidate_mask=np.stack(candidate_masks, axis=0),
        target_y=np.asarray(target_y, dtype=np.int64),
        pollution_y=np.stack(pollution_targets, axis=0),
        fragmentation_y=np.asarray(fragmentation_targets, dtype=np.float32),
        shortlist_sizes=np.asarray(shortlist_sizes, dtype=np.int64),
        num_decisions=num_decisions,
        num_training_examples=len(target_y),
        dropped_missing_gold=dropped_missing_gold,
    )


def save_joint_training_data(data: JointRouterData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        aggregate_x=data.aggregate_x,
        candidate_x=data.candidate_x,
        candidate_mask=data.candidate_mask,
        target_y=data.target_y,
        pollution_y=data.pollution_y,
        fragmentation_y=data.fragmentation_y,
        shortlist_sizes=data.shortlist_sizes,
        metadata=np.asarray(
            [data.num_decisions, data.num_training_examples, data.dropped_missing_gold],
            dtype=np.int64,
        ),
    )


def load_joint_training_data(input_path: Path) -> JointRouterData:
    payload = np.load(input_path, allow_pickle=False)
    metadata = payload["metadata"].astype(np.int64)
    target_y = payload["target_y"].astype(np.int64)
    candidate_mask = payload["candidate_mask"].astype(bool)
    if "pollution_y" in payload:
        pollution_y = payload["pollution_y"].astype(np.float32)
    else:
        pollution_y = np.ones(candidate_mask.shape, dtype=np.float32)
        link_rows = target_y > 0
        row_indices = np.where(link_rows)[0]
        pollution_y[row_indices, target_y[link_rows] - 1] = 0.0
    if "fragmentation_y" in payload:
        fragmentation_y = payload["fragmentation_y"].astype(np.float32)
    else:
        fragmentation_y = (target_y > 0).astype(np.float32)
    return JointRouterData(
        aggregate_x=payload["aggregate_x"].astype(np.float32),
        candidate_x=payload["candidate_x"].astype(np.float32),
        candidate_mask=candidate_mask,
        target_y=target_y,
        pollution_y=pollution_y,
        fragmentation_y=fragmentation_y,
        shortlist_sizes=payload["shortlist_sizes"].astype(np.int64),
        num_decisions=int(metadata[0]),
        num_training_examples=int(metadata[1]),
        dropped_missing_gold=int(metadata[2]),
    )


def _fit_standardizer(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _fit_masked_standardizer(train_x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_rows = train_x[mask]
    return _fit_standardizer(valid_rows)


def _legacy_feature_indices(current_dim: int, target_dim: int) -> list[int] | None:
    # Old checkpoints were trained before anchor-specific candidate features were inserted
    # into the middle of the memory feature vector. To keep those checkpoints usable for
    # evaluation and calibration, project current features back to the legacy layout.
    old_memory_feature_idx = list(range(17)) + [20, 21, 22, 23]
    if current_dim == 24 and target_dim == 21:
        return old_memory_feature_idx
    if current_dim == 29 and target_dim == 26:
        return old_memory_feature_idx + list(range(24, 29))
    if current_dim == 54 and target_dim == 48:
        return old_memory_feature_idx + list(range(24, 30)) + [30 + idx for idx in old_memory_feature_idx]
    if current_dim == 39 and target_dim == 36:
        return old_memory_feature_idx + list(range(24, 29)) + list(range(29, 39))
    if current_dim == 78 and target_dim == 72:
        return (
            old_memory_feature_idx
            + list(range(24, 30))
            + [30 + idx for idx in old_memory_feature_idx]
            + list(range(54, 78))
        )
    return None


def _align_feature_last_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = int(x.shape[-1])
    if current_dim == target_dim:
        return x
    legacy_idx = _legacy_feature_indices(current_dim, target_dim)
    if legacy_idx is not None:
        return x[..., legacy_idx]
    if current_dim > target_dim:
        return x[..., :target_dim]
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - current_dim)
    return np.pad(x, pad_width, mode="constant")


def _apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    aligned_x = _align_feature_last_dim(np.asarray(x, dtype=np.float32), int(mean.shape[0]))
    return ((aligned_x - mean) / std).astype(np.float32)


class JointMemoryRouter(nn.Module):
    def __init__(
        self,
        aggregate_dim: int,
        candidate_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.15,
        enable_risk_heads: bool = False,
    ):
        super().__init__()
        self.enable_risk_heads = enable_risk_heads

        self.maturity_weights = nn.Parameter(torch.tensor([
            0.35,
            0.30,
            0.10,
            0.20,
            0.25,
            0.15,
            0.15,
            0.15,
            -0.20,
            -0.10,
            -0.05,
        ], dtype=torch.float32))

        actual_candidate_dim = candidate_dim - 11 + 1  # 35 - 11 + 1 = 25

        self.candidate_encoder = nn.Sequential(
            nn.Linear(actual_candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.candidate_scorer = nn.Linear(hidden_dim, 1)
        self.create_head = nn.Sequential(
            nn.Linear(aggregate_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.novelty_head = nn.Sequential(
            nn.Linear(aggregate_dim + actual_candidate_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if enable_risk_heads:
            self.pollution_head = nn.Linear(hidden_dim, 1)
            self.fragmentation_head = nn.Sequential(
                nn.Linear(aggregate_dim + actual_candidate_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward_with_risk(
        self,
        aggregate_x: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, top_k, candidate_dim = candidate_x.shape

        raw_maturity_features = candidate_x[:, :, -11:]
        maturity_scores = torch.sigmoid((raw_maturity_features * self.maturity_weights).sum(dim=-1))

        candidate_x_with_maturity = torch.cat([
            candidate_x[:, :, :24],
            maturity_scores.unsqueeze(-1),
        ], dim=-1)

        flat_candidate = candidate_x_with_maturity.view(batch_size * top_k, 25)
        hidden = self.candidate_encoder(flat_candidate).view(batch_size, top_k, -1)
        candidate_logits = self.candidate_scorer(hidden).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e4)

        float_mask = candidate_mask.unsqueeze(-1).float()
        mean_hidden = (hidden * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp(min=1.0)
        max_hidden = hidden.masked_fill(~candidate_mask.unsqueeze(-1), -1e4).max(dim=1).values
        max_hidden = torch.where(torch.isfinite(max_hidden), max_hidden, torch.zeros_like(max_hidden))
        top_hidden = hidden[:, 0, :]
        top_candidate_x = candidate_x_with_maturity[:, 0, :]

        create_input = torch.cat([aggregate_x, mean_hidden, max_hidden], dim=1)
        base_create_logit = self.create_head(create_input).squeeze(-1)
        novelty_input = torch.cat([aggregate_x, top_candidate_x, top_hidden], dim=1)
        novelty_adjustment = self.novelty_head(novelty_input).squeeze(-1)
        create_logit = base_create_logit + novelty_adjustment
        joint_logits = torch.cat([create_logit.unsqueeze(1), candidate_logits], dim=1)
        pollution_logits = None
        fragmentation_logit = None
        if self.enable_risk_heads:
            pollution_logits = self.pollution_head(hidden).squeeze(-1)
            pollution_logits = pollution_logits.masked_fill(~candidate_mask, -1e4)
            fragmentation_logit = self.fragmentation_head(novelty_input).squeeze(-1)
        return (
            create_logit,
            candidate_logits,
            joint_logits,
            novelty_adjustment,
            pollution_logits,
            fragmentation_logit,
        )

    def forward(
        self,
        aggregate_x: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        create_logit, candidate_logits, joint_logits, novelty_adjustment, _, _ = self.forward_with_risk(
            aggregate_x,
            candidate_x,
            candidate_mask,
        )
        return create_logit, candidate_logits, joint_logits, novelty_adjustment


@dataclass
class TrainedJointRouter:
    model: JointMemoryRouter
    aggregate_mean: np.ndarray
    aggregate_std: np.ndarray
    candidate_mean: np.ndarray
    candidate_std: np.ndarray
    hidden_dim: int
    dropout: float
    history: list[dict[str, float]]
    selection_history: list[dict[str, float]]
    best_selection: dict[str, float] | None
    enable_risk_heads: bool = False
    risk_config: dict[str, float] = field(default_factory=dict)


def save_trained_router(trained: TrainedJointRouter, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": trained.model.state_dict(),
        "aggregate_mean": trained.aggregate_mean,
        "aggregate_std": trained.aggregate_std,
        "candidate_mean": trained.candidate_mean,
        "candidate_std": trained.candidate_std,
        "hidden_dim": trained.hidden_dim,
        "dropout": trained.dropout,
        "enable_risk_heads": trained.enable_risk_heads,
        "risk_config": trained.risk_config or {},
        "history": trained.history,
        "selection_history": trained.selection_history,
        "best_selection": trained.best_selection,
    }
    torch.save(payload, output_path)


def _ensure_numpy_pickle_compat() -> None:
    # Older checkpoints can reference numpy._core when serialized under newer numpy builds.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


def load_trained_router(checkpoint_path: Path, device: str) -> TrainedJointRouter:
    _ensure_numpy_pickle_compat()
    payload = torch.load(checkpoint_path, map_location=device)
    aggregate_mean = np.asarray(payload["aggregate_mean"], dtype=np.float32)
    aggregate_std = np.asarray(payload["aggregate_std"], dtype=np.float32)
    candidate_mean = np.asarray(payload["candidate_mean"], dtype=np.float32)
    candidate_std = np.asarray(payload["candidate_std"], dtype=np.float32)
    hidden_dim = int(payload["hidden_dim"])
    dropout = float(payload["dropout"])
    enable_risk_heads = bool(payload.get("enable_risk_heads", False))
    model = JointMemoryRouter(
        aggregate_dim=int(aggregate_mean.shape[0]),
        candidate_dim=int(candidate_mean.shape[0]),
        hidden_dim=hidden_dim,
        dropout=dropout,
        enable_risk_heads=enable_risk_heads,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return TrainedJointRouter(
        model=model,
        aggregate_mean=aggregate_mean,
        aggregate_std=aggregate_std,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        hidden_dim=hidden_dim,
        dropout=dropout,
        enable_risk_heads=enable_risk_heads,
        risk_config=dict(payload.get("risk_config", {})),
        history=list(payload.get("history", [])),
        selection_history=list(payload.get("selection_history", [])),
        best_selection=payload.get("best_selection"),
    )


def _prediction_trace_row(
    prediction: EpisodePrediction,
    record: StreamingMentionRecord,
    num_slots_before: int,
    shortlist_size: int,
    create_logit: float | None,
    applied_create_bias: float | None,
    best_link_logit: float | None,
    second_logit: float | None,
    novelty_adjustment: float | None,
    best_candidate_size_before: int | None,
    best_candidate_doc_count_before: int | None,
    best_candidate_age_before: int | None,
    best_candidate_memory_id: str | None,
    chosen_slot_size_before: int | None,
    chosen_slot_doc_count_before: int | None,
    chosen_slot_age_before: int | None,
    age_bucket: str | None,
    doc_bucket: str | None,
    memory_bucket: str | None,
    margin_bucket: str | None,
    best_candidate_maturity: dict[str, float] | None = None,
    chosen_slot_maturity: dict[str, float] | None = None,
    visual_reliability: float | None = None,
    visual_update_accepted: bool | None = None,
    pollution_risk: float | None = None,
    fragmentation_risk: float | None = None,
) -> dict[str, object]:
    row = asdict(prediction)
    row.update(
        {
            "episode_id": record.episode_id,
            "stream_index": int(record.stream_index),
            "doc_id": record.doc_id,
            "topic": record.topic,
            "num_slots_before": int(num_slots_before),
            "shortlist_size": int(shortlist_size),
            "create_logit": None if create_logit is None else float(create_logit),
            "applied_create_bias": None if applied_create_bias is None else float(applied_create_bias),
            "best_link_logit": None if best_link_logit is None else float(best_link_logit),
            "second_logit": None if second_logit is None else float(second_logit),
            "novelty_adjustment": None if novelty_adjustment is None else float(novelty_adjustment),
            "best_candidate_size_before": best_candidate_size_before,
            "best_candidate_doc_count_before": best_candidate_doc_count_before,
            "best_candidate_age_before": best_candidate_age_before,
            "best_candidate_memory_id": best_candidate_memory_id,
            "chosen_slot_size_before": chosen_slot_size_before,
            "chosen_slot_doc_count_before": chosen_slot_doc_count_before,
            "chosen_slot_age_before": chosen_slot_age_before,
            "age_bucket": age_bucket,
            "doc_bucket": doc_bucket,
            "memory_bucket": memory_bucket,
            "margin_bucket": margin_bucket,
            "best_candidate_maturity_score": (
                None if best_candidate_maturity is None else best_candidate_maturity.get("maturity_score")
            ),
            "best_candidate_maturity_bucket": (
                None if best_candidate_maturity is None else best_candidate_maturity.get("maturity_bucket")
            ),
            "best_candidate_visual_coverage": (
                None if best_candidate_maturity is None else best_candidate_maturity.get("visual_coverage")
            ),
            "best_candidate_cross_modal_agreement": (
                None if best_candidate_maturity is None else best_candidate_maturity.get("cross_modal_agreement")
            ),
            "chosen_slot_maturity_score": (
                None if chosen_slot_maturity is None else chosen_slot_maturity.get("maturity_score")
            ),
            "chosen_slot_maturity_bucket": (
                None if chosen_slot_maturity is None else chosen_slot_maturity.get("maturity_bucket")
            ),
            "visual_reliability": None if visual_reliability is None else float(visual_reliability),
            "visual_update_accepted": visual_update_accepted,
            "pollution_risk": None if pollution_risk is None else float(pollution_risk),
            "fragmentation_risk": None if fragmentation_risk is None else float(fragmentation_risk),
        }
    )
    return row


def summarize_risk_diagnostics(trace_rows: list[dict[str, object]]) -> dict[str, object]:
    if not trace_rows:
        return {
            "num_mentions": 0,
            "false_link_rate": 0.0,
            "false_create_rate": 0.0,
            "maturity_bucket_metrics": {},
            "visual_gate": {},
        }

    false_links = 0
    predicted_links = 0
    false_creates = 0
    predicted_creates = 0
    gold_links = 0
    gold_creates = 0
    bucket_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    visual_available = 0
    visual_update_accepted = 0
    visual_reliability_values: list[float] = []
    pollution_risk_values: list[float] = []
    fragmentation_risk_values: list[float] = []
    pollution_risk_false_link_values: list[float] = []
    pollution_risk_true_link_values: list[float] = []
    fragmentation_risk_false_create_values: list[float] = []
    fragmentation_risk_true_create_values: list[float] = []
    pollution_auc_labels: list[int] = []
    pollution_auc_scores: list[float] = []
    fragmentation_auc_labels: list[int] = []
    fragmentation_auc_scores: list[float] = []

    for row in trace_rows:
        gold_action = str(row.get("gold_action", ""))
        predicted_action = str(row.get("predicted_action", ""))
        if gold_action == "LINK":
            gold_links += 1
        if gold_action == "CREATE":
            gold_creates += 1
        if predicted_action == "LINK":
            predicted_links += 1
            false_links += int(gold_action != "LINK")
        if predicted_action == "CREATE":
            predicted_creates += 1
            false_creates += int(gold_action != "CREATE")

        bucket_value = row.get("chosen_slot_maturity_bucket")
        if bucket_value is None:
            bucket_value = row.get("best_candidate_maturity_bucket")
        bucket = "none" if bucket_value is None else str(int(float(bucket_value)))
        bucket_rows[bucket].append(row)

        reliability = row.get("visual_reliability")
        if reliability is not None:
            visual_reliability_values.append(float(reliability))
            visual_available += 1
            visual_update_accepted += int(bool(row.get("visual_update_accepted")))
        pollution_risk = row.get("pollution_risk")
        if pollution_risk is not None:
            pollution_value = float(pollution_risk)
            pollution_risk_values.append(pollution_value)
            if predicted_action == "LINK" and gold_action != "LINK":
                pollution_risk_false_link_values.append(pollution_value)
                pollution_auc_labels.append(1)
                pollution_auc_scores.append(pollution_value)
            elif predicted_action == "LINK":
                pollution_risk_true_link_values.append(pollution_value)
                pollution_auc_labels.append(0)
                pollution_auc_scores.append(pollution_value)
        fragmentation_risk = row.get("fragmentation_risk")
        if fragmentation_risk is not None:
            fragmentation_value = float(fragmentation_risk)
            fragmentation_risk_values.append(fragmentation_value)
            if predicted_action == "CREATE" and gold_action != "CREATE":
                fragmentation_risk_false_create_values.append(fragmentation_value)
                fragmentation_auc_labels.append(1)
                fragmentation_auc_scores.append(fragmentation_value)
            elif predicted_action == "CREATE":
                fragmentation_risk_true_create_values.append(fragmentation_value)
                fragmentation_auc_labels.append(0)
                fragmentation_auc_scores.append(fragmentation_value)

    bucket_metrics: dict[str, dict[str, float]] = {}
    for bucket, rows in bucket_rows.items():
        bucket_false_links = sum(
            int(row.get("predicted_action") == "LINK" and row.get("gold_action") != "LINK")
            for row in rows
        )
        bucket_predicted_links = sum(int(row.get("predicted_action") == "LINK") for row in rows)
        bucket_false_creates = sum(
            int(row.get("predicted_action") == "CREATE" and row.get("gold_action") != "CREATE")
            for row in rows
        )
        bucket_predicted_creates = sum(int(row.get("predicted_action") == "CREATE") for row in rows)
        bucket_metrics[bucket] = {
            "num_mentions": float(len(rows)),
            "false_link_rate": float(bucket_false_links / max(bucket_predicted_links, 1)),
            "false_create_rate": float(bucket_false_creates / max(bucket_predicted_creates, 1)),
            "predicted_link_rate": float(bucket_predicted_links / max(len(rows), 1)),
            "predicted_create_rate": float(bucket_predicted_creates / max(len(rows), 1)),
        }

    return {
        "num_mentions": len(trace_rows),
        "gold_link_rate": float(gold_links / max(len(trace_rows), 1)),
        "gold_create_rate": float(gold_creates / max(len(trace_rows), 1)),
        "predicted_link_rate": float(predicted_links / max(len(trace_rows), 1)),
        "predicted_create_rate": float(predicted_creates / max(len(trace_rows), 1)),
        "false_link_rate": float(false_links / max(predicted_links, 1)),
        "false_create_rate": float(false_creates / max(predicted_creates, 1)),
        "maturity_bucket_metrics": bucket_metrics,
        "visual_gate": {
            "visual_available_trace_rows": visual_available,
            "accepted_updates": visual_update_accepted,
            "accept_rate": float(visual_update_accepted / max(visual_available, 1)),
            "mean_visual_reliability": (
                float(np.mean(visual_reliability_values)) if visual_reliability_values else 0.0
            ),
        },
        "risk_heads": {
            "num_pollution_scores": len(pollution_risk_values),
            "mean_pollution_risk": float(np.mean(pollution_risk_values)) if pollution_risk_values else 0.0,
            "mean_pollution_risk_false_link": (
                float(np.mean(pollution_risk_false_link_values)) if pollution_risk_false_link_values else 0.0
            ),
            "mean_pollution_risk_true_link": (
                float(np.mean(pollution_risk_true_link_values)) if pollution_risk_true_link_values else 0.0
            ),
            "pollution_risk_auc_on_predicted_links": _binary_auc(
                pollution_auc_labels,
                pollution_auc_scores,
            ),
            "num_fragmentation_scores": len(fragmentation_risk_values),
            "mean_fragmentation_risk": (
                float(np.mean(fragmentation_risk_values)) if fragmentation_risk_values else 0.0
            ),
            "mean_fragmentation_risk_false_create": (
                float(np.mean(fragmentation_risk_false_create_values))
                if fragmentation_risk_false_create_values
                else 0.0
            ),
            "mean_fragmentation_risk_true_create": (
                float(np.mean(fragmentation_risk_true_create_values))
                if fragmentation_risk_true_create_values
                else 0.0
            ),
            "fragmentation_risk_auc_on_predicted_creates": _binary_auc(
                fragmentation_auc_labels,
                fragmentation_auc_scores,
            ),
        },
    }


def _masked_margin_loss(
    candidate_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_y: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    link_mask = target_y > 0
    if not torch.any(link_mask):
        return candidate_logits.new_tensor(0.0)

    link_logits = candidate_logits[link_mask]
    link_valid_mask = candidate_mask[link_mask]
    gold_index = target_y[link_mask] - 1
    gold_logits = link_logits.gather(1, gold_index.unsqueeze(1)).squeeze(1)
    losses = F.relu(margin - gold_logits.unsqueeze(1) + link_logits)

    candidate_ids = torch.arange(candidate_logits.size(1), device=candidate_logits.device).unsqueeze(0)
    negative_mask = link_valid_mask & (candidate_ids != gold_index.unsqueeze(1))
    if not torch.any(negative_mask):
        return candidate_logits.new_tensor(0.0)

    losses = losses * negative_mask.float()
    return losses.sum() / negative_mask.float().sum().clamp(min=1.0)


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels:
        return None
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    num_positive = int(label_array.sum())
    num_negative = int(label_array.size - num_positive)
    if num_positive == 0 or num_negative == 0:
        return None

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.zeros_like(score_array, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[label_array == 1].sum())
    auc = (positive_rank_sum - num_positive * (num_positive + 1) / 2.0) / (
        num_positive * num_negative
    )
    return float(auc)


def train_joint_router(
    data: JointRouterData,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    dropout: float,
    margin_weight: float,
    create_weight: float,
    ranking_margin: float,
    dev_records: list[StreamingMentionRecord] | None = None,
    top_k: int = 5,
    selection_biases: list[float] | None = None,
    eval_every: int = 0,
    ambiguity_alpha: float = 0.0,
    single_doc_alpha: float = 0.0,
    attachment_guard_weight: float = 0.0,
    attachment_guard_margin: float = 0.2,
    attachment_guard_ambiguity_alpha: float = 0.0,
    attachment_guard_single_doc_alpha: float = 0.0,
    enable_risk_heads: bool = False,
    pollution_risk_weight: float = 0.0,
    fragmentation_risk_weight: float = 0.0,
    risk_margin_weight: float = 0.0,
    pollution_risk_scale: float = 0.0,
    fragmentation_risk_scale: float = 0.0,
    selection_prefix_records: list[StreamingMentionRecord] | None = None,
    selection_protocol: str = "cold",
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    seed: int = 42,
) -> TrainedJointRouter:
    sample_weight = _build_example_weights(
        aggregate_x=data.aggregate_x,
        candidate_x=data.candidate_x,
        target_y=data.target_y,
        ambiguity_alpha=ambiguity_alpha,
        single_doc_alpha=single_doc_alpha,
    )
    attachment_guard_weight_vec = _build_attachment_guard_weights(
        aggregate_x=data.aggregate_x,
        candidate_x=data.candidate_x,
        target_y=data.target_y,
        ambiguity_guard_alpha=attachment_guard_ambiguity_alpha,
        single_doc_guard_alpha=attachment_guard_single_doc_alpha,
    )
    aggregate_mean, aggregate_std = _fit_standardizer(data.aggregate_x)
    candidate_mean, candidate_std = _fit_masked_standardizer(data.candidate_x, data.candidate_mask)

    aggregate_x = _apply_standardizer(data.aggregate_x, aggregate_mean, aggregate_std)
    candidate_x = _apply_standardizer(data.candidate_x, candidate_mean, candidate_std)

    dataset = TensorDataset(
        torch.from_numpy(aggregate_x),
        torch.from_numpy(candidate_x),
        torch.from_numpy(data.candidate_mask),
        torch.from_numpy(data.target_y),
        torch.from_numpy(data.pollution_y.astype(np.float32)),
        torch.from_numpy(data.fragmentation_y.astype(np.float32)),
        torch.from_numpy(sample_weight),
        torch.from_numpy(attachment_guard_weight_vec),
    )
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=data_generator,
    )

    model = JointMemoryRouter(
        aggregate_dim=aggregate_x.shape[1],
        candidate_dim=candidate_x.shape[2],
        hidden_dim=hidden_dim,
        dropout=dropout,
        enable_risk_heads=enable_risk_heads,
    ).to(device)

    create_targets = (data.target_y == 0).astype(np.float32)
    create_positive_rate = float(create_targets.mean())
    create_pos_weight = (1.0 - create_positive_rate) / max(create_positive_rate, 1e-6)
    create_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([create_pos_weight], device=device, dtype=torch.float32)
        , reduction="none"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_metric = -1.0
    stale = 0
    patience = 8
    history: list[dict[str, float]] = []
    selection_history: list[dict[str, float]] = []
    best_selection: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_create = 0.0
        total_margin = 0.0
        total_guard = 0.0
        total_pollution = 0.0
        total_fragmentation = 0.0
        total_risk_margin = 0.0
        total_novelty_abs = 0.0
        total_items = 0

        for (
            aggregate_batch,
            candidate_batch,
            mask_batch,
            target_batch,
            pollution_batch,
            fragmentation_batch,
            weight_batch,
            guard_weight_batch,
        ) in loader:
            aggregate_batch = aggregate_batch.to(device)
            candidate_batch = candidate_batch.to(device)
            mask_batch = mask_batch.to(device)
            target_batch = target_batch.to(device)
            pollution_batch = pollution_batch.to(device)
            fragmentation_batch = fragmentation_batch.to(device)
            weight_batch = weight_batch.to(device)
            guard_weight_batch = guard_weight_batch.to(device)

            (
                create_logit,
                candidate_logits,
                joint_logits,
                novelty_adjustment,
                pollution_logits,
                fragmentation_logit,
            ) = model.forward_with_risk(
                aggregate_batch,
                candidate_batch,
                mask_batch,
            )

            ce_loss = F.cross_entropy(joint_logits, target_batch, reduction="none")
            ce_loss = (ce_loss * weight_batch).sum() / weight_batch.sum().clamp(min=1.0)
            create_loss = create_criterion(create_logit, (target_batch == 0).float())
            create_loss = (create_loss * weight_batch).sum() / weight_batch.sum().clamp(min=1.0)
            margin_loss = _masked_margin_loss(candidate_logits, mask_batch, target_batch, ranking_margin)
            if attachment_guard_weight > 0.0:
                top_link_logit = candidate_logits.max(dim=1).values
                has_candidates = mask_batch.any(dim=1).float()
                guard_margin_gap = create_logit - top_link_logit
                guard_loss_raw = F.relu(attachment_guard_margin - guard_margin_gap)
                effective_guard_weight = guard_weight_batch * has_candidates
                guard_denom = effective_guard_weight.sum().clamp(min=1.0)
                guard_loss = (guard_loss_raw * effective_guard_weight).sum() / guard_denom
            else:
                guard_loss = torch.zeros((), device=device)
            if enable_risk_heads and pollution_logits is not None and fragmentation_logit is not None:
                pollution_loss_raw = F.binary_cross_entropy_with_logits(
                    pollution_logits,
                    pollution_batch,
                    reduction="none",
                )
                pollution_weight = mask_batch.float() * weight_batch.unsqueeze(1)
                pollution_loss = (pollution_loss_raw * pollution_weight).sum() / pollution_weight.sum().clamp(min=1.0)
                fragmentation_loss_raw = F.binary_cross_entropy_with_logits(
                    fragmentation_logit,
                    fragmentation_batch,
                    reduction="none",
                )
                fragmentation_loss = (fragmentation_loss_raw * weight_batch).sum() / weight_batch.sum().clamp(min=1.0)
                link_mask = target_batch > 0
                if torch.any(link_mask):
                    link_rows = torch.where(link_mask)[0]
                    gold_index = target_batch[link_mask] - 1
                    gold_risk = pollution_logits[link_rows, gold_index]
                    candidate_ids = torch.arange(pollution_logits.size(1), device=device).unsqueeze(0)
                    negative_mask = mask_batch[link_mask] & (candidate_ids != gold_index.unsqueeze(1))
                    risk_margin_raw = F.relu(0.25 + gold_risk.unsqueeze(1) - pollution_logits[link_mask])
                    risk_margin = (risk_margin_raw * negative_mask.float()).sum() / negative_mask.float().sum().clamp(min=1.0)
                else:
                    risk_margin = torch.zeros((), device=device)
            else:
                pollution_loss = torch.zeros((), device=device)
                fragmentation_loss = torch.zeros((), device=device)
                risk_margin = torch.zeros((), device=device)
            loss = (
                ce_loss
                + create_weight * create_loss
                + margin_weight * margin_loss
                + attachment_guard_weight * guard_loss
                + pollution_risk_weight * pollution_loss
                + fragmentation_risk_weight * fragmentation_loss
                + risk_margin_weight * risk_margin
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_items = aggregate_batch.size(0)
            total_loss += float(loss.item()) * batch_items
            total_ce += float(ce_loss.item()) * batch_items
            total_create += float(create_loss.item()) * batch_items
            total_margin += float(margin_loss.item()) * batch_items
            total_guard += float(guard_loss.item()) * batch_items
            total_pollution += float(pollution_loss.item()) * batch_items
            total_fragmentation += float(fragmentation_loss.item()) * batch_items
            total_risk_margin += float(risk_margin.item()) * batch_items
            total_novelty_abs += float(novelty_adjustment.detach().abs().mean().item()) * batch_items
            total_items += batch_items

        epoch_row = {
            "epoch": float(epoch),
            "loss": total_loss / max(total_items, 1),
            "ce_loss": total_ce / max(total_items, 1),
            "create_loss": total_create / max(total_items, 1),
            "margin_loss": total_margin / max(total_items, 1),
            "guard_loss": total_guard / max(total_items, 1),
            "pollution_loss": total_pollution / max(total_items, 1),
            "fragmentation_loss": total_fragmentation / max(total_items, 1),
            "risk_margin_loss": total_risk_margin / max(total_items, 1),
            "novelty_abs": total_novelty_abs / max(total_items, 1),
        }
        history.append(epoch_row)
        print(
            f"epoch={epoch} loss={epoch_row['loss']:.4f} "
            f"ce={epoch_row['ce_loss']:.4f} create={epoch_row['create_loss']:.4f} "
            f"margin={epoch_row['margin_loss']:.4f} guard={epoch_row['guard_loss']:.4f} "
            f"pollution={epoch_row['pollution_loss']:.4f} "
            f"fragment={epoch_row['fragmentation_loss']:.4f}",
            flush=True,
        )

        improved = False
        use_dev_selection = bool(dev_records and selection_biases and eval_every > 0 and epoch % eval_every == 0)
        if use_dev_selection:
            model.eval()
            temp_trained = TrainedJointRouter(
                model=model,
                aggregate_mean=aggregate_mean,
                aggregate_std=aggregate_std,
                candidate_mean=candidate_mean,
                candidate_std=candidate_std,
                hidden_dim=hidden_dim,
                dropout=dropout,
                enable_risk_heads=enable_risk_heads,
                risk_config={
                    "pollution_risk_weight": float(pollution_risk_weight),
                    "fragmentation_risk_weight": float(fragmentation_risk_weight),
                    "risk_margin_weight": float(risk_margin_weight),
                    "pollution_risk_scale": float(pollution_risk_scale),
                    "fragmentation_risk_scale": float(fragmentation_risk_scale),
                },
                history=history,
                selection_history=selection_history,
                best_selection=best_selection,
            )
            best_epoch_metric = -1.0
            best_epoch_bias = 0.0
            for create_bias in selection_biases:
                if selection_protocol == "continuation" and selection_prefix_records is not None:
                    dev_metrics = run_joint_router_continuation(
                        prefix_records=selection_prefix_records,
                        eval_records=dev_records,
                        trained=temp_trained,
                        top_k=top_k,
                        device=device,
                        create_bias=create_bias,
                        retrieval_config=retrieval_config,
                        pollution_risk_scale=pollution_risk_scale,
                        fragmentation_risk_scale=fragmentation_risk_scale,
                    )
                else:
                    dev_metrics = run_joint_router(
                        dev_records,
                        trained=temp_trained,
                        top_k=top_k,
                        device=device,
                        create_bias=create_bias,
                        retrieval_config=retrieval_config,
                        pollution_risk_scale=pollution_risk_scale,
                        fragmentation_risk_scale=fragmentation_risk_scale,
                    )
                if dev_metrics["pair_f1"] > best_epoch_metric:
                    best_epoch_metric = dev_metrics["pair_f1"]
                    best_epoch_bias = create_bias
            selection_row = {
                "epoch": float(epoch),
                "best_dev_pair_f1": float(best_epoch_metric),
                "best_create_bias": float(best_epoch_bias),
            }
            selection_history.append(selection_row)
            print(
                f"selection epoch={epoch} dev_f1={best_epoch_metric:.4f} "
                f"bias={best_epoch_bias:+.2f}",
                flush=True,
            )
            if best_epoch_metric > best_metric + 1e-6:
                best_metric = best_epoch_metric
                best_state = copy.deepcopy(model.state_dict())
                best_selection = selection_row
                improved = True
        elif epoch_row["loss"] + 1e-6 < best_loss:
            best_loss = epoch_row["loss"]
            best_state = copy.deepcopy(model.state_dict())
            improved = True
        if improved:
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"early_stop epoch={epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainedJointRouter(
        model=model,
        aggregate_mean=aggregate_mean,
        aggregate_std=aggregate_std,
        candidate_mean=candidate_mean,
        candidate_std=candidate_std,
        hidden_dim=hidden_dim,
        dropout=dropout,
        enable_risk_heads=enable_risk_heads,
        risk_config={
            "pollution_risk_weight": float(pollution_risk_weight),
            "fragmentation_risk_weight": float(fragmentation_risk_weight),
            "risk_margin_weight": float(risk_margin_weight),
            "pollution_risk_scale": float(pollution_risk_scale),
            "fragmentation_risk_scale": float(fragmentation_risk_scale),
        },
        history=history,
        selection_history=selection_history,
        best_selection=best_selection,
    )


def _predict_joint_logits(
    trained: TrainedJointRouter,
    aggregate_x: np.ndarray,
    candidate_x: np.ndarray,
    candidate_mask: np.ndarray,
    device: str,
) -> np.ndarray:
    standardized_aggregate = _apply_standardizer(aggregate_x, trained.aggregate_mean, trained.aggregate_std)
    standardized_candidate = _apply_standardizer(candidate_x, trained.candidate_mean, trained.candidate_std)
    with torch.no_grad():
        aggregate_tensor = torch.from_numpy(standardized_aggregate).to(device)
        candidate_tensor = torch.from_numpy(standardized_candidate).to(device)
        mask_tensor = torch.from_numpy(candidate_mask).to(device)
        _, _, joint_logits, _ = trained.model(aggregate_tensor, candidate_tensor, mask_tensor)
        return joint_logits.detach().cpu().numpy()


def run_joint_router(
    records: list[StreamingMentionRecord],
    trained: TrainedJointRouter,
    top_k: int,
    device: str,
    create_bias: float,
    single_doc_create_bias: float = 0.0,
    stale_create_bias: float = 0.0,
    stale_age_threshold: int = 8,
    ambiguous_create_bias: float = 0.0,
    ambiguity_threshold: float = 0.25,
    state_aware_bias_config: dict[str, Any] | None = None,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    pollution_risk_scale: float = 0.0,
    fragmentation_risk_scale: float = 0.0,
) -> dict[str, float]:
    metrics, _ = run_joint_router_detailed(
        records=records,
        trained=trained,
        top_k=top_k,
        device=device,
        create_bias=create_bias,
        single_doc_create_bias=single_doc_create_bias,
        stale_create_bias=stale_create_bias,
        stale_age_threshold=stale_age_threshold,
        ambiguous_create_bias=ambiguous_create_bias,
        ambiguity_threshold=ambiguity_threshold,
        state_aware_bias_config=state_aware_bias_config,
        retrieval_config=retrieval_config,
        pollution_risk_scale=pollution_risk_scale,
        fragmentation_risk_scale=fragmentation_risk_scale,
    )
    return metrics


def _run_joint_router_sequence(
    records: list[StreamingMentionRecord],
    trained: TrainedJointRouter,
    top_k: int,
    device: str,
    create_bias: float,
    single_doc_create_bias: float = 0.0,
    stale_create_bias: float = 0.0,
    stale_age_threshold: int = 8,
    ambiguous_create_bias: float = 0.0,
    ambiguity_threshold: float = 0.25,
    state_aware_bias_config: dict[str, Any] | None = None,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    pollution_risk_scale: float = 0.0,
    fragmentation_risk_scale: float = 0.0,
    state: dict[str, list[tuple[str, MemorySlot]]] | None = None,
    next_ids: dict[str, int] | None = None,
    collect_predictions: bool = True,
) -> tuple[dict[str, float] | None, list[dict[str, object]], dict[str, list[tuple[str, MemorySlot]]], dict[str, int]]:
    episodes = _group_by_episode(records)
    all_predictions: list[EpisodePrediction] = []
    trace_rows: list[dict[str, object]] = []
    state = state if state is not None else defaultdict(list)
    next_ids = next_ids if next_ids is not None else defaultdict(int)
    normalized_state_aware_bias_config = _normalize_state_aware_bias_config(state_aware_bias_config)

    for episode_id, episode_records in episodes.items():
        slots = state[episode_id]
        for record in episode_records:
            if not slots:
                next_ids[episode_id] += 1
                slot = _new_slot(f"mem_{next_ids[episode_id]}", record)
                slots.append((slot.memory_id, slot))
                if collect_predictions:
                    prediction = EpisodePrediction(
                        mention_id=record.mention_id,
                        gold_cluster=record.gold_cluster,
                        predicted_memory_id=slot.memory_id,
                        gold_action=record.gold_action,
                        predicted_action="CREATE",
                        score=1.0,
                        episode_id=record.episode_id,
                    )
                    all_predictions.append(prediction)
                    trace_rows.append(
                        _prediction_trace_row(
                            prediction=prediction,
                            record=record,
                            num_slots_before=0,
                            shortlist_size=0,
                            create_logit=None,
                            applied_create_bias=None,
                            best_link_logit=None,
                            second_logit=None,
                            novelty_adjustment=None,
                            best_candidate_size_before=None,
                            best_candidate_doc_count_before=None,
                            best_candidate_age_before=None,
                            best_candidate_memory_id=None,
                            chosen_slot_size_before=None,
                            chosen_slot_doc_count_before=None,
                            chosen_slot_age_before=None,
                            age_bucket=None,
                            doc_bucket=None,
                            memory_bucket=None,
                            margin_bucket=None,
                            best_candidate_maturity=None,
                            chosen_slot_maturity=None,
                            visual_reliability=None,
                            visual_update_accepted=None,
                        )
                    )
                continue

            shortlist = _ranked_shortlist(
                slots,
                record,
                top_k,
                retrieval_config=retrieval_config,
            )
            aggregate_x = _aggregate_shortlist_features(shortlist, num_slots=len(slots)).reshape(1, -1)
            candidate_x = np.zeros((1, top_k, shortlist_candidate_row_dim()), dtype=np.float32)
            candidate_mask = np.zeros((1, top_k), dtype=bool)
            memory_ids: list[str] = []
            for idx, (memory_id, features, retrieval) in enumerate(shortlist):
                candidate_x[0, idx] = _candidate_row(slots, shortlist, memory_id, features, retrieval)
                candidate_mask[0, idx] = True
                memory_ids.append(memory_id)
            best_candidate_slot = slots[0][1]
            if memory_ids:
                best_candidate_slot = next(slot for memory_id, slot in slots if memory_id == memory_ids[0])

            standardized_aggregate = _apply_standardizer(
                aggregate_x, trained.aggregate_mean, trained.aggregate_std
            )
            standardized_candidate = _apply_standardizer(
                candidate_x, trained.candidate_mean, trained.candidate_std
            )
            with torch.no_grad():
                aggregate_tensor = torch.from_numpy(standardized_aggregate).to(device)
                candidate_tensor = torch.from_numpy(standardized_candidate).to(device)
                mask_tensor = torch.from_numpy(candidate_mask).to(device)
                (
                    _create_logit_batch,
                    _candidate_logits_batch,
                    joint_logits_batch,
                    novelty_adjustment_batch,
                    pollution_logits_batch,
                    fragmentation_logit_batch,
                ) = trained.model.forward_with_risk(
                    aggregate_tensor, candidate_tensor, mask_tensor
                )
                joint_logits = joint_logits_batch.detach().cpu().numpy()[0]
                novelty_adjustment = float(novelty_adjustment_batch.detach().cpu().numpy()[0])
                pollution_risks = None
                fragmentation_risk = None
                if pollution_logits_batch is not None:
                    pollution_risks = torch.sigmoid(pollution_logits_batch).detach().cpu().numpy()[0]
                if fragmentation_logit_batch is not None:
                    fragmentation_risk = float(torch.sigmoid(fragmentation_logit_batch).detach().cpu().numpy()[0])
            raw_create_logit = float(joint_logits[0])
            valid_link_logits = [float(joint_logits[idx + 1]) for idx in range(len(memory_ids))]
            best_link_logit = max(valid_link_logits) if valid_link_logits else None
            best_candidate_size_before = len(best_candidate_slot.mentions)
            best_candidate_doc_count_before = len(best_candidate_slot.doc_ids)
            best_candidate_age_before = max(record.stream_index - best_candidate_slot.last_stream_index, 0)
            best_candidate_memory_id = memory_ids[0] if memory_ids else None
            best_candidate_maturity = (
                memory_maturity_features(best_candidate_slot, record)
                if best_candidate_memory_id is not None
                else None
            )
            applied_create_bias, bucket_snapshot = _compose_create_bias(
                create_bias=create_bias,
                single_doc_create_bias=single_doc_create_bias,
                stale_create_bias=stale_create_bias,
                stale_age_threshold=stale_age_threshold,
                ambiguous_create_bias=ambiguous_create_bias,
                ambiguity_threshold=ambiguity_threshold,
                num_slots_before=len(slots),
                best_candidate_doc_count_before=best_candidate_doc_count_before,
                best_candidate_age_before=best_candidate_age_before,
                raw_create_logit=raw_create_logit,
                best_link_logit=best_link_logit,
                state_aware_bias_config=normalized_state_aware_bias_config,
            )
            joint_logits[0] += applied_create_bias
            if pollution_risks is not None and pollution_risk_scale != 0.0:
                for idx in range(len(memory_ids)):
                    joint_logits[idx + 1] -= float(pollution_risk_scale) * float(pollution_risks[idx])
            if fragmentation_risk is not None and fragmentation_risk_scale != 0.0:
                joint_logits[0] -= float(fragmentation_risk_scale) * float(fragmentation_risk)
            decision_index = int(np.argmax(joint_logits))
            create_logit = float(joint_logits[0])
            sorted_logits = sorted([float(value) for value in joint_logits[: len(memory_ids) + 1]], reverse=True)
            second_logit = sorted_logits[1] if len(sorted_logits) > 1 else None

            if decision_index == 0:
                next_ids[episode_id] += 1
                slot = _new_slot(f"mem_{next_ids[episode_id]}", record)
                slots.append((slot.memory_id, slot))
                if collect_predictions:
                    prediction = EpisodePrediction(
                        mention_id=record.mention_id,
                        gold_cluster=record.gold_cluster,
                        predicted_memory_id=slot.memory_id,
                        gold_action=record.gold_action,
                        predicted_action="CREATE",
                        score=float(joint_logits[0]),
                        episode_id=record.episode_id,
                    )
                    all_predictions.append(prediction)
                    trace_rows.append(
                        _prediction_trace_row(
                            prediction=prediction,
                            record=record,
                            num_slots_before=len(slots) - 1,
                            shortlist_size=len(memory_ids),
                            create_logit=create_logit,
                            applied_create_bias=applied_create_bias,
                            best_link_logit=best_link_logit,
                            second_logit=second_logit,
                            novelty_adjustment=novelty_adjustment,
                            best_candidate_size_before=best_candidate_size_before,
                            best_candidate_doc_count_before=best_candidate_doc_count_before,
                            best_candidate_age_before=best_candidate_age_before,
                            best_candidate_memory_id=best_candidate_memory_id,
                            chosen_slot_size_before=None,
                            chosen_slot_doc_count_before=None,
                            chosen_slot_age_before=None,
                            age_bucket=bucket_snapshot["age_bucket"],
                            doc_bucket=bucket_snapshot["doc_bucket"],
                            memory_bucket=bucket_snapshot["memory_bucket"],
                            margin_bucket=bucket_snapshot["margin_bucket"],
                            best_candidate_maturity=best_candidate_maturity,
                            chosen_slot_maturity=None,
                            visual_reliability=None,
                            visual_update_accepted=None,
                            pollution_risk=None,
                            fragmentation_risk=fragmentation_risk,
                        )
                    )
                continue

            chosen_memory_id = memory_ids[decision_index - 1]
            chosen_slot = next(slot for memory_id, slot in slots if memory_id == chosen_memory_id)
            chosen_slot_size_before = len(chosen_slot.mentions)
            chosen_slot_doc_count_before = len(chosen_slot.doc_ids)
            chosen_slot_age_before = max(record.stream_index - chosen_slot.last_stream_index, 0)
            chosen_slot_maturity = memory_maturity_features(chosen_slot, record)
            pollution_risk = None
            if pollution_risks is not None and 0 <= decision_index - 1 < len(pollution_risks):
                pollution_risk = float(pollution_risks[decision_index - 1])
            visual_reliability = visual_reliability_score(chosen_slot, record) if record.has_visual else None
            visual_update_accepted = None
            if record.has_visual:
                visual_update_accepted = bool(retrieval_config.visual_update_mode != "off")
                if retrieval_config.visual_update_mode == "reliability_gated":
                    visual_update_accepted = bool(
                        visual_reliability is not None
                        and visual_reliability >= retrieval_config.visual_reliability_threshold
                    )
            _update_slot(chosen_slot, record, retrieval_config=retrieval_config)
            if collect_predictions:
                prediction = EpisodePrediction(
                    mention_id=record.mention_id,
                    gold_cluster=record.gold_cluster,
                    predicted_memory_id=chosen_memory_id,
                    gold_action=record.gold_action,
                    predicted_action="LINK",
                    score=float(joint_logits[decision_index]),
                    episode_id=record.episode_id,
                )
                all_predictions.append(prediction)
                trace_rows.append(
                    _prediction_trace_row(
                        prediction=prediction,
                        record=record,
                        num_slots_before=len(slots),
                        shortlist_size=len(memory_ids),
                        create_logit=create_logit,
                        applied_create_bias=applied_create_bias,
                        best_link_logit=best_link_logit,
                        second_logit=second_logit,
                        novelty_adjustment=novelty_adjustment,
                        best_candidate_size_before=best_candidate_size_before,
                        best_candidate_doc_count_before=best_candidate_doc_count_before,
                        best_candidate_age_before=best_candidate_age_before,
                        best_candidate_memory_id=best_candidate_memory_id,
                        chosen_slot_size_before=chosen_slot_size_before,
                        chosen_slot_doc_count_before=chosen_slot_doc_count_before,
                        chosen_slot_age_before=chosen_slot_age_before,
                        age_bucket=bucket_snapshot["age_bucket"],
                        doc_bucket=bucket_snapshot["doc_bucket"],
                        memory_bucket=bucket_snapshot["memory_bucket"],
                        margin_bucket=bucket_snapshot["margin_bucket"],
                        best_candidate_maturity=best_candidate_maturity,
                        chosen_slot_maturity=chosen_slot_maturity,
                        visual_reliability=visual_reliability,
                        visual_update_accepted=visual_update_accepted,
                        pollution_risk=pollution_risk,
                        fragmentation_risk=fragmentation_risk,
                    )
                )

    metrics = evaluate_episode_predictions(all_predictions) if collect_predictions else None
    return metrics, trace_rows, state, next_ids


def run_joint_router_continuation(
    prefix_records: list[StreamingMentionRecord],
    eval_records: list[StreamingMentionRecord],
    trained: TrainedJointRouter,
    top_k: int,
    device: str,
    create_bias: float,
    single_doc_create_bias: float = 0.0,
    stale_create_bias: float = 0.0,
    stale_age_threshold: int = 8,
    ambiguous_create_bias: float = 0.0,
    ambiguity_threshold: float = 0.25,
    state_aware_bias_config: dict[str, Any] | None = None,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    pollution_risk_scale: float = 0.0,
    fragmentation_risk_scale: float = 0.0,
) -> dict[str, float]:
    metrics, _, _, _ = run_joint_router_continuation_detailed(
        prefix_records=prefix_records,
        eval_records=eval_records,
        trained=trained,
        top_k=top_k,
        device=device,
        create_bias=create_bias,
        single_doc_create_bias=single_doc_create_bias,
        stale_create_bias=stale_create_bias,
        stale_age_threshold=stale_age_threshold,
        ambiguous_create_bias=ambiguous_create_bias,
        ambiguity_threshold=ambiguity_threshold,
        state_aware_bias_config=state_aware_bias_config,
        retrieval_config=retrieval_config,
        pollution_risk_scale=pollution_risk_scale,
        fragmentation_risk_scale=fragmentation_risk_scale,
    )
    return metrics


def run_joint_router_continuation_detailed(
    prefix_records: list[StreamingMentionRecord],
    eval_records: list[StreamingMentionRecord],
    trained: TrainedJointRouter,
    top_k: int,
    device: str,
    create_bias: float,
    single_doc_create_bias: float = 0.0,
    stale_create_bias: float = 0.0,
    stale_age_threshold: int = 8,
    ambiguous_create_bias: float = 0.0,
    ambiguity_threshold: float = 0.25,
    state_aware_bias_config: dict[str, Any] | None = None,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    pollution_risk_scale: float = 0.0,
    fragmentation_risk_scale: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, list[tuple[str, MemorySlot]]], dict[str, int]]:
    _, _, state, next_ids = _run_joint_router_sequence(
        records=prefix_records,
        trained=trained,
        top_k=top_k,
        device=device,
        create_bias=create_bias,
        single_doc_create_bias=single_doc_create_bias,
        stale_create_bias=stale_create_bias,
        stale_age_threshold=stale_age_threshold,
        ambiguous_create_bias=ambiguous_create_bias,
        ambiguity_threshold=ambiguity_threshold,
        state_aware_bias_config=state_aware_bias_config,
        retrieval_config=retrieval_config,
        pollution_risk_scale=pollution_risk_scale,
        fragmentation_risk_scale=fragmentation_risk_scale,
        state=None,
        next_ids=None,
        collect_predictions=False,
    )
    metrics, trace_rows, state, next_ids = _run_joint_router_sequence(
        records=eval_records,
        trained=trained,
        top_k=top_k,
        device=device,
        create_bias=create_bias,
        single_doc_create_bias=single_doc_create_bias,
        stale_create_bias=stale_create_bias,
        stale_age_threshold=stale_age_threshold,
        ambiguous_create_bias=ambiguous_create_bias,
        ambiguity_threshold=ambiguity_threshold,
        state_aware_bias_config=state_aware_bias_config,
        retrieval_config=retrieval_config,
        pollution_risk_scale=pollution_risk_scale,
        fragmentation_risk_scale=fragmentation_risk_scale,
        state=state,
        next_ids=next_ids,
        collect_predictions=True,
    )
    return metrics, trace_rows, state, next_ids


def run_joint_router_detailed(
    records: list[StreamingMentionRecord],
    trained: TrainedJointRouter,
    top_k: int,
    device: str,
    create_bias: float,
    single_doc_create_bias: float = 0.0,
    stale_create_bias: float = 0.0,
    stale_age_threshold: int = 8,
    ambiguous_create_bias: float = 0.0,
    ambiguity_threshold: float = 0.25,
    state_aware_bias_config: dict[str, Any] | None = None,
    retrieval_config: RetrievalConfig = BASE_RETRIEVAL_CONFIG,
    pollution_risk_scale: float = 0.0,
    fragmentation_risk_scale: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    metrics, trace_rows, _, _ = _run_joint_router_sequence(
        records=records,
        trained=trained,
        top_k=top_k,
        device=device,
        create_bias=create_bias,
        single_doc_create_bias=single_doc_create_bias,
        stale_create_bias=stale_create_bias,
        stale_age_threshold=stale_age_threshold,
        ambiguous_create_bias=ambiguous_create_bias,
        ambiguity_threshold=ambiguity_threshold,
        state_aware_bias_config=state_aware_bias_config,
        retrieval_config=retrieval_config,
        pollution_risk_scale=pollution_risk_scale,
        fragmentation_risk_scale=fragmentation_risk_scale,
        state=None,
        next_ids=None,
        collect_predictions=True,
    )
    return metrics, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--margin-weight", type=float, default=0.2)
    parser.add_argument("--create-weight", type=float, default=0.35)
    parser.add_argument("--ranking-margin", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--selection-biases", default="-0.5,-0.25,0.0,0.25,0.5")
    parser.add_argument("--create-biases", default="-0.75,-0.5,-0.25,0.0,0.25,0.5")
    parser.add_argument("--single-doc-create-biases", default="0.0")
    parser.add_argument("--stale-create-biases", default="0.0")
    parser.add_argument("--stale-age-threshold", type=int, default=8)
    parser.add_argument("--ambiguous-create-biases", default="0.0")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.25)
    parser.add_argument("--ambiguity-alpha", type=float, default=0.0)
    parser.add_argument("--single-doc-alpha", type=float, default=0.0)
    parser.add_argument("--attachment-guard-weight", type=float, default=0.0)
    parser.add_argument("--attachment-guard-margin", type=float, default=0.2)
    parser.add_argument("--attachment-guard-ambiguity-alpha", type=float, default=0.0)
    parser.add_argument("--attachment-guard-single-doc-alpha", type=float, default=0.0)
    parser.add_argument("--enable-risk-heads", action="store_true")
    parser.add_argument("--pollution-risk-weight", type=float, default=0.0)
    parser.add_argument("--fragmentation-risk-weight", type=float, default=0.0)
    parser.add_argument("--risk-margin-weight", type=float, default=0.0)
    parser.add_argument("--pollution-risk-scale", type=float, default=0.0)
    parser.add_argument("--fragmentation-risk-scale", type=float, default=0.0)
    parser.add_argument("--retrieval-mode", choices=["base", "anchor"], default="base")
    parser.add_argument("--anchor-title-bonus", type=float, default=0.45)
    parser.add_argument("--anchor-context-bonus", type=float, default=0.10)
    parser.add_argument("--anchor-token-bonus", type=float, default=0.08)
    parser.add_argument("--disable-maturity-signals", action="store_true")
    parser.add_argument("--disable-visual-signals", action="store_true")
    parser.add_argument("--disable-anchor-history", action="store_true")
    parser.add_argument("--visual-update-mode", choices=["always", "off", "reliability_gated"], default="always")
    parser.add_argument("--visual-reliability-threshold", type=float, default=0.0)
    parser.add_argument("--visual-update-rate", type=float, default=1.0)
    parser.add_argument("--evaluation-protocol", choices=["cold", "continuation"], default="cold")
    parser.add_argument("--final-sweep-mode", choices=["full", "selection_only", "none"], default="full")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--training-data-cache-path", default="")
    parser.add_argument("--overwrite-training-data-cache", action="store_true")
    parser.add_argument("--training-data-progress-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    set_random_seed(args.seed)
    benchmark_dir = Path(args.benchmark_dir)
    print(f"loading_records benchmark_dir={benchmark_dir}", flush=True)
    load_start = time.time()
    train_records = load_records(benchmark_dir / "train_stream_records.pkl")
    dev_records = load_records(benchmark_dir / "dev_stream_records.pkl")
    test_records = load_records(benchmark_dir / "test_stream_records.pkl")
    print(
        f"loaded_records train={len(train_records)} dev={len(dev_records)} "
        f"test={len(test_records)} elapsed_sec={time.time() - load_start:.1f}",
        flush=True,
    )
    retrieval_config = RetrievalConfig(
        mode=args.retrieval_mode,
        anchor_title_bonus=args.anchor_title_bonus,
        anchor_context_bonus=args.anchor_context_bonus,
        anchor_token_bonus=args.anchor_token_bonus,
        disable_maturity_signals=args.disable_maturity_signals,
        disable_visual_signals=args.disable_visual_signals,
        disable_anchor_history=args.disable_anchor_history,
        visual_update_mode=args.visual_update_mode,
        visual_reliability_threshold=args.visual_reliability_threshold,
        visual_update_rate=args.visual_update_rate,
    )

    cache_path = Path(args.training_data_cache_path) if args.training_data_cache_path else None
    if cache_path is not None and cache_path.exists() and not args.overwrite_training_data_cache:
        print(f"loading_training_data_cache path={cache_path}", flush=True)
        data_start = time.time()
        data = load_joint_training_data(cache_path)
        print(
            f"loaded_training_data_cache examples={data.num_training_examples} "
            f"decisions={data.num_decisions} dropped_missing_gold={data.dropped_missing_gold} "
            f"elapsed_sec={time.time() - data_start:.1f}",
            flush=True,
        )
    else:
        print("building_joint_training_data", flush=True)
        data_start = time.time()
        data = build_joint_training_data(
            train_records,
            top_k=args.top_k,
            retrieval_config=retrieval_config,
            progress_every=args.training_data_progress_every,
        )
        print(
            f"built_joint_training_data examples={data.num_training_examples} "
            f"decisions={data.num_decisions} dropped_missing_gold={data.dropped_missing_gold} "
            f"aggregate_dim={data.aggregate_x.shape[1]} candidate_dim={data.candidate_x.shape[2]} "
            f"elapsed_sec={time.time() - data_start:.1f}",
            flush=True,
        )
        if cache_path is not None:
            print(f"saving_training_data_cache path={cache_path}", flush=True)
            save_joint_training_data(data, cache_path)
    trained = train_joint_router(
        data=data,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        margin_weight=args.margin_weight,
        create_weight=args.create_weight,
        ranking_margin=args.ranking_margin,
        dev_records=dev_records,
        top_k=args.top_k,
        selection_biases=[float(item) for item in args.selection_biases.split(",") if item],
        eval_every=args.eval_every,
        ambiguity_alpha=args.ambiguity_alpha,
        single_doc_alpha=args.single_doc_alpha,
        attachment_guard_weight=args.attachment_guard_weight,
        attachment_guard_margin=args.attachment_guard_margin,
        attachment_guard_ambiguity_alpha=args.attachment_guard_ambiguity_alpha,
        attachment_guard_single_doc_alpha=args.attachment_guard_single_doc_alpha,
        enable_risk_heads=args.enable_risk_heads,
        pollution_risk_weight=args.pollution_risk_weight,
        fragmentation_risk_weight=args.fragmentation_risk_weight,
        risk_margin_weight=args.risk_margin_weight,
        pollution_risk_scale=args.pollution_risk_scale,
        fragmentation_risk_scale=args.fragmentation_risk_scale,
        selection_prefix_records=train_records if args.evaluation_protocol == "continuation" else None,
        selection_protocol=args.evaluation_protocol,
        retrieval_config=retrieval_config,
        seed=args.seed,
    )
    if args.checkpoint_path:
        save_trained_router(trained, Path(args.checkpoint_path))
        print(f"saved_checkpoint path={args.checkpoint_path}", flush=True)

    bias_values = [float(item) for item in args.create_biases.split(",") if item]
    single_doc_bias_values = [float(item) for item in args.single_doc_create_biases.split(",") if item]
    stale_bias_values = [float(item) for item in args.stale_create_biases.split(",") if item]
    ambiguous_bias_values = [float(item) for item in args.ambiguous_create_biases.split(",") if item]
    results = []
    best = None

    def _risk_kwargs() -> dict[str, float]:
        return {
            "pollution_risk_scale": args.pollution_risk_scale,
            "fragmentation_risk_scale": args.fragmentation_risk_scale,
        }

    if args.final_sweep_mode == "full":
        for create_bias in bias_values:
            for single_doc_create_bias in single_doc_bias_values:
                for stale_create_bias in stale_bias_values:
                    for ambiguous_create_bias in ambiguous_bias_values:
                        if args.evaluation_protocol == "continuation":
                            dev_metrics = run_joint_router_continuation(
                                prefix_records=train_records,
                                eval_records=dev_records,
                                trained=trained,
                                top_k=args.top_k,
                                device=args.device,
                                create_bias=create_bias,
                                single_doc_create_bias=single_doc_create_bias,
                                stale_create_bias=stale_create_bias,
                                stale_age_threshold=args.stale_age_threshold,
                                ambiguous_create_bias=ambiguous_create_bias,
                                ambiguity_threshold=args.ambiguity_threshold,
                                retrieval_config=retrieval_config,
                                **_risk_kwargs(),
                            )
                            test_metrics = run_joint_router_continuation(
                                prefix_records=train_records + dev_records,
                                eval_records=test_records,
                                trained=trained,
                                top_k=args.top_k,
                                device=args.device,
                                create_bias=create_bias,
                                single_doc_create_bias=single_doc_create_bias,
                                stale_create_bias=stale_create_bias,
                                stale_age_threshold=args.stale_age_threshold,
                                ambiguous_create_bias=ambiguous_create_bias,
                                ambiguity_threshold=args.ambiguity_threshold,
                                retrieval_config=retrieval_config,
                                **_risk_kwargs(),
                            )
                        else:
                            dev_metrics = run_joint_router(
                                dev_records,
                                trained=trained,
                                top_k=args.top_k,
                                device=args.device,
                                create_bias=create_bias,
                                single_doc_create_bias=single_doc_create_bias,
                                stale_create_bias=stale_create_bias,
                                stale_age_threshold=args.stale_age_threshold,
                                ambiguous_create_bias=ambiguous_create_bias,
                                ambiguity_threshold=args.ambiguity_threshold,
                                retrieval_config=retrieval_config,
                                **_risk_kwargs(),
                            )
                            test_metrics = run_joint_router(
                                test_records,
                                trained=trained,
                                top_k=args.top_k,
                                device=args.device,
                                create_bias=create_bias,
                                single_doc_create_bias=single_doc_create_bias,
                                stale_create_bias=stale_create_bias,
                                stale_age_threshold=args.stale_age_threshold,
                                ambiguous_create_bias=ambiguous_create_bias,
                                ambiguity_threshold=args.ambiguity_threshold,
                                retrieval_config=retrieval_config,
                                **_risk_kwargs(),
                            )
                        row = {
                            "top_k": args.top_k,
                            "create_bias": create_bias,
                            "single_doc_create_bias": single_doc_create_bias,
                            "stale_create_bias": stale_create_bias,
                            "stale_age_threshold": args.stale_age_threshold,
                            "ambiguous_create_bias": ambiguous_create_bias,
                            "ambiguity_threshold": args.ambiguity_threshold,
                            "dev": dev_metrics,
                            "test": test_metrics,
                        }
                        results.append(row)
                        print(
                            f"bias={create_bias:+.2f} single_doc={single_doc_create_bias:+.2f} "
                            f"stale={stale_create_bias:+.2f} ambiguous={ambiguous_create_bias:+.2f} "
                            f"dev_f1={dev_metrics['pair_f1']:.4f} test_f1={test_metrics['pair_f1']:.4f}",
                            flush=True,
                        )
                        if best is None or dev_metrics["pair_f1"] > best["dev"]["pair_f1"]:
                            best = row
    elif args.final_sweep_mode == "selection_only":
        create_bias = float(trained.best_selection["best_create_bias"]) if trained.best_selection else bias_values[0]
        single_doc_create_bias = 0.0
        stale_create_bias = 0.0
        ambiguous_create_bias = 0.0
        if args.evaluation_protocol == "continuation":
            dev_metrics = run_joint_router_continuation(
                prefix_records=train_records,
                eval_records=dev_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=create_bias,
                single_doc_create_bias=single_doc_create_bias,
                stale_create_bias=stale_create_bias,
                stale_age_threshold=args.stale_age_threshold,
                ambiguous_create_bias=ambiguous_create_bias,
                ambiguity_threshold=args.ambiguity_threshold,
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
            test_metrics = run_joint_router_continuation(
                prefix_records=train_records + dev_records,
                eval_records=test_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=create_bias,
                single_doc_create_bias=single_doc_create_bias,
                stale_create_bias=stale_create_bias,
                stale_age_threshold=args.stale_age_threshold,
                ambiguous_create_bias=ambiguous_create_bias,
                ambiguity_threshold=args.ambiguity_threshold,
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
        else:
            dev_metrics = run_joint_router(
                dev_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=create_bias,
                single_doc_create_bias=single_doc_create_bias,
                stale_create_bias=stale_create_bias,
                stale_age_threshold=args.stale_age_threshold,
                ambiguous_create_bias=ambiguous_create_bias,
                ambiguity_threshold=args.ambiguity_threshold,
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
            test_metrics = run_joint_router(
                test_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=create_bias,
                single_doc_create_bias=single_doc_create_bias,
                stale_create_bias=stale_create_bias,
                stale_age_threshold=args.stale_age_threshold,
                ambiguous_create_bias=ambiguous_create_bias,
                ambiguity_threshold=args.ambiguity_threshold,
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
        best = {
            "top_k": args.top_k,
            "create_bias": create_bias,
            "single_doc_create_bias": single_doc_create_bias,
            "stale_create_bias": stale_create_bias,
            "stale_age_threshold": args.stale_age_threshold,
            "ambiguous_create_bias": ambiguous_create_bias,
            "ambiguity_threshold": args.ambiguity_threshold,
            "dev": dev_metrics,
            "test": test_metrics,
        }
        results.append(best)
        print(
            f"selection_only bias={create_bias:+.2f} single_doc={single_doc_create_bias:+.2f} "
            f"stale={stale_create_bias:+.2f} ambiguous={ambiguous_create_bias:+.2f} "
            f"dev_f1={dev_metrics['pair_f1']:.4f} test_f1={test_metrics['pair_f1']:.4f}",
            flush=True,
        )
    else:
        print("final_sweep skipped", flush=True)

    payload = {
        "model": "gpu_joint_novelty_calibrated_router",
        "device": args.device,
        "top_k": args.top_k,
        "seed": args.seed,
        "num_decisions": data.num_decisions,
        "training_examples": data.num_training_examples,
        "dropped_missing_gold": data.dropped_missing_gold,
        "retrieval_coverage": data.num_training_examples / max(data.num_decisions, 1),
        "create_rate": float((data.target_y == 0).mean()),
        "history": trained.history,
        "selection_history": trained.selection_history,
        "best_selection": trained.best_selection,
        "ambiguity_alpha": args.ambiguity_alpha,
        "single_doc_alpha": args.single_doc_alpha,
        "attachment_guard_weight": args.attachment_guard_weight,
        "attachment_guard_margin": args.attachment_guard_margin,
        "attachment_guard_ambiguity_alpha": args.attachment_guard_ambiguity_alpha,
        "attachment_guard_single_doc_alpha": args.attachment_guard_single_doc_alpha,
        "enable_risk_heads": args.enable_risk_heads,
        "risk_config": {
            "pollution_risk_weight": args.pollution_risk_weight,
            "fragmentation_risk_weight": args.fragmentation_risk_weight,
            "risk_margin_weight": args.risk_margin_weight,
            "pollution_risk_scale": args.pollution_risk_scale,
            "fragmentation_risk_scale": args.fragmentation_risk_scale,
        },
        "evaluation_protocol": args.evaluation_protocol,
        "final_sweep_mode": args.final_sweep_mode,
        "retrieval_config": {
            "mode": retrieval_config.mode,
            "anchor_title_bonus": retrieval_config.anchor_title_bonus,
            "anchor_context_bonus": retrieval_config.anchor_context_bonus,
            "anchor_token_bonus": retrieval_config.anchor_token_bonus,
            "disable_maturity_signals": retrieval_config.disable_maturity_signals,
            "disable_visual_signals": retrieval_config.disable_visual_signals,
            "disable_anchor_history": retrieval_config.disable_anchor_history,
            "visual_update_mode": retrieval_config.visual_update_mode,
            "visual_reliability_threshold": retrieval_config.visual_reliability_threshold,
            "visual_update_rate": retrieval_config.visual_update_rate,
        },
        "memory_candidate_feature_names": memory_candidate_feature_names(),
        "memory_candidate_feature_groups": memory_candidate_feature_group_indices(),
        "single_doc_create_biases": single_doc_bias_values,
        "stale_create_biases": stale_bias_values,
        "ambiguous_create_biases": ambiguous_bias_values,
        "stale_age_threshold": args.stale_age_threshold,
        "ambiguity_threshold": args.ambiguity_threshold,
        "best": best,
        "results": results,
    }
    save_json(payload, Path(args.output_path))
    if args.prediction_dir and best is not None:
        prediction_dir = Path(args.prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        best_bias = float(best["create_bias"])
        if args.evaluation_protocol == "continuation":
            dev_metrics, dev_predictions, _, _ = run_joint_router_continuation_detailed(
                prefix_records=train_records,
                eval_records=dev_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=best_bias,
                single_doc_create_bias=float(best.get("single_doc_create_bias", 0.0)),
                stale_create_bias=float(best.get("stale_create_bias", 0.0)),
                stale_age_threshold=int(best.get("stale_age_threshold", args.stale_age_threshold)),
                ambiguous_create_bias=float(best.get("ambiguous_create_bias", 0.0)),
                ambiguity_threshold=float(best.get("ambiguity_threshold", args.ambiguity_threshold)),
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
            test_metrics, test_predictions, _, _ = run_joint_router_continuation_detailed(
                prefix_records=train_records + dev_records,
                eval_records=test_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=best_bias,
                single_doc_create_bias=float(best.get("single_doc_create_bias", 0.0)),
                stale_create_bias=float(best.get("stale_create_bias", 0.0)),
                stale_age_threshold=int(best.get("stale_age_threshold", args.stale_age_threshold)),
                ambiguous_create_bias=float(best.get("ambiguous_create_bias", 0.0)),
                ambiguity_threshold=float(best.get("ambiguity_threshold", args.ambiguity_threshold)),
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
        else:
            dev_metrics, dev_predictions = run_joint_router_detailed(
                dev_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=best_bias,
                single_doc_create_bias=float(best.get("single_doc_create_bias", 0.0)),
                stale_create_bias=float(best.get("stale_create_bias", 0.0)),
                stale_age_threshold=int(best.get("stale_age_threshold", args.stale_age_threshold)),
                ambiguous_create_bias=float(best.get("ambiguous_create_bias", 0.0)),
                ambiguity_threshold=float(best.get("ambiguity_threshold", args.ambiguity_threshold)),
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
            test_metrics, test_predictions = run_joint_router_detailed(
                test_records,
                trained=trained,
                top_k=args.top_k,
                device=args.device,
                create_bias=best_bias,
                single_doc_create_bias=float(best.get("single_doc_create_bias", 0.0)),
                stale_create_bias=float(best.get("stale_create_bias", 0.0)),
                stale_age_threshold=int(best.get("stale_age_threshold", args.stale_age_threshold)),
                ambiguous_create_bias=float(best.get("ambiguous_create_bias", 0.0)),
                ambiguity_threshold=float(best.get("ambiguity_threshold", args.ambiguity_threshold)),
                retrieval_config=retrieval_config,
                **_risk_kwargs(),
            )
        save_json(
            {
                "split": "dev",
                "metrics": dev_metrics,
                "create_bias": best_bias,
                "evaluation_protocol": args.evaluation_protocol,
                "retrieval_config": payload["retrieval_config"],
                "risk_diagnostics": summarize_risk_diagnostics(dev_predictions),
                "predictions": dev_predictions,
            },
            prediction_dir / "dev_predictions.json",
        )
        save_json(
            {
                "split": "test",
                "metrics": test_metrics,
                "create_bias": best_bias,
                "evaluation_protocol": args.evaluation_protocol,
                "retrieval_config": payload["retrieval_config"],
                "risk_diagnostics": summarize_risk_diagnostics(test_predictions),
                "predictions": test_predictions,
            },
            prediction_dir / "test_predictions.json",
        )
    print(payload["best"], flush=True)


if __name__ == "__main__":
    main()
