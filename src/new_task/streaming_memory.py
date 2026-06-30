from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

DOC_ID_PATTERN = re.compile(r"(?P<topic>\d+)_(?P<doc>\d+)")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
SECTION_PREFIX = "section:"
CONTEXT_PREFIX = "context:"
EVENT_TITLE_PREFIX = "event_title:"


def _to_numpy(vector: object, dim: int) -> np.ndarray:
    if isinstance(vector, np.ndarray):
        arr = vector.astype(np.float32, copy=False)
    elif hasattr(vector, "detach"):
        arr = vector.detach().cpu().numpy().astype(np.float32, copy=False)
    elif vector is None:
        arr = np.zeros(dim, dtype=np.float32)
    else:
        arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == dim:
        return arr
    if arr.size > dim:
        return arr[:dim]
    padded = np.zeros(dim, dtype=np.float32)
    padded[: arr.size] = arr
    return padded


def _infer_dim(vector: object, fallback_dim: int) -> int:
    if isinstance(vector, np.ndarray):
        arr = vector
    elif hasattr(vector, "detach"):
        arr = vector.detach().cpu().numpy()
    elif vector is None:
        return fallback_dim
    else:
        arr = np.asarray(vector)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return int(arr.size) if arr.size > 0 else fallback_dim


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _doc_order(doc_id: str) -> tuple[int, int, str]:
    match = DOC_ID_PATTERN.match(doc_id)
    if not match:
        return (10**9, 10**9, doc_id)
    return (
        int(match.group("topic")),
        int(match.group("doc")),
        doc_id,
    )


def _normalize_anchor_text(text: str) -> str:
    text = (
        text.strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def _anchor_tokens(text: str) -> tuple[str, ...]:
    normalized = _normalize_anchor_text(text)
    if not normalized:
        return ()
    tokens = [token for token in NON_WORD_RE.split(normalized) if token]
    return tuple(tokens)


@lru_cache(maxsize=200000)
def _parse_bert_doc_fields(bert_doc: str) -> tuple[str, tuple[str, ...], str]:
    section = ""
    context_titles: list[str] = []
    event_title = ""
    for raw_line in bert_doc.splitlines():
        line = raw_line.strip()
        if line.startswith(SECTION_PREFIX):
            section = line[len(SECTION_PREFIX) :].strip()
        elif line.startswith(CONTEXT_PREFIX):
            context = line[len(CONTEXT_PREFIX) :].strip()
            if context:
                context_titles = [part.strip() for part in context.split("|") if part.strip()]
        elif line.startswith(EVENT_TITLE_PREFIX):
            event_title = line[len(EVENT_TITLE_PREFIX) :].strip()
    return section, tuple(context_titles), event_title


def record_anchor_title(record: "StreamingMentionRecord") -> str:
    _section, _context_titles, event_title = _parse_bert_doc_fields(record.bert_doc)
    if event_title:
        return _normalize_anchor_text(event_title)
    if record.lemma:
        return _normalize_anchor_text(record.lemma)
    return _normalize_anchor_text(record.mention_text)


def record_context_titles(record: "StreamingMentionRecord") -> tuple[str, ...]:
    _section, context_titles, _event_title = _parse_bert_doc_fields(record.bert_doc)
    return tuple(_normalize_anchor_text(item) for item in context_titles if item.strip())


def record_anchor_token_set(record: "StreamingMentionRecord") -> set[str]:
    title_tokens = set(_anchor_tokens(record_anchor_title(record)))
    if title_tokens:
        return title_tokens
    return set(_anchor_tokens(record.mention_text))


def anchor_overlap_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union <= 0:
        return 0.0
    return float(intersection / union)


@dataclass
class StreamingMentionRecord:
    mention_id: str
    split: str
    episode_id: str
    stream_index: int
    doc_id: str
    sentence_id: int
    mention_text: str
    bert_doc: str
    topic: str
    predicted_topic: str
    gold_cluster: str
    gold_action: str
    has_visual: bool
    mention_vector: np.ndarray
    visual_vector: np.ndarray
    lemma: str

    def to_storage_dict(self) -> dict:
        payload = asdict(self)
        payload["mention_vector"] = self.mention_vector.astype(np.float32)
        payload["visual_vector"] = self.visual_vector.astype(np.float32)
        return payload


def build_streaming_memory_records(
    mention_map: dict[str, dict],
    visual_features: dict[str, np.ndarray],
    split: str,
    text_dim: int = 300,
    visual_dim: int = 512,
) -> list[StreamingMentionRecord]:
    split_mentions = [
        (mention_id, item)
        for mention_id, item in mention_map.items()
        if item.get("men_type") == "evt" and item.get("split") == split
    ]
    split_mentions.sort(
        key=lambda item: (
            str(item[1].get("topic", "")),
            _doc_order(str(item[1].get("doc_id", ""))),
            int(item[1].get("sentence_id", 0) or 0),
            item[0],
        )
    )

    seen_clusters: dict[str, set[str]] = defaultdict(set)
    episode_counts: dict[str, int] = defaultdict(int)
    records: list[StreamingMentionRecord] = []
    for mention_id, item in split_mentions:
        episode_id = str(item.get("topic", ""))
        episode_counts[episode_id] += 1
        gold_cluster = str(item.get("gold_cluster", ""))
        if gold_cluster and gold_cluster not in seen_clusters[episode_id]:
            gold_action = "CREATE"
            seen_clusters[episode_id].add(gold_cluster)
        else:
            gold_action = "LINK"

        visual_vector = visual_features.get(mention_id)
        has_visual = visual_vector is not None
        record = StreamingMentionRecord(
            mention_id=mention_id,
            split=split,
            episode_id=episode_id,
            stream_index=episode_counts[episode_id] - 1,
            doc_id=str(item.get("doc_id", "")),
            sentence_id=int(item.get("sentence_id", 0) or 0),
            mention_text=str(item.get("mention_text", "")),
            bert_doc=str(item.get("bert_doc", "")),
            topic=str(item.get("topic", "")),
            predicted_topic=str(item.get("predicted_topic", "")),
            gold_cluster=gold_cluster,
            gold_action=gold_action,
            has_visual=has_visual,
            mention_vector=_to_numpy(item.get("mention_vector"), text_dim),
            visual_vector=_to_numpy(visual_vector, visual_dim),
            lemma=str(item.get("lemma", "")),
        )
        records.append(record)
    return records


@dataclass
class EpisodePrediction:
    mention_id: str
    gold_cluster: str
    predicted_memory_id: str
    gold_action: str
    predicted_action: str
    score: float
    episode_id: str = ""


@dataclass
class MemorySlot:
    memory_id: str
    mentions: list[str]
    prototype_text: np.ndarray
    prototype_visual: np.ndarray
    prototype_lemma: str
    text_history: list[np.ndarray]
    visual_history: list[np.ndarray]
    text_count: int = 0
    visual_count: int = 0
    visual_reliability_sum: float = 0.0
    last_visual_reliability: float = 0.0
    first_stream_index: int = 0
    last_stream_index: int = 0
    last_doc_id: str = ""
    doc_ids: set[str] = field(default_factory=set)
    anchor_titles: set[str] = field(default_factory=set)
    context_titles: set[str] = field(default_factory=set)
    anchor_token_set: set[str] = field(default_factory=set)


def new_memory_slot(memory_id: str, record: StreamingMentionRecord) -> MemorySlot:
    anchor_title = record_anchor_title(record)
    context_titles = set(record_context_titles(record))
    anchor_token_set = record_anchor_token_set(record)
    return MemorySlot(
        memory_id=memory_id,
        mentions=[record.mention_id],
        prototype_text=record.mention_vector.copy(),
        prototype_visual=record.visual_vector.copy(),
        prototype_lemma=record.lemma,
        text_history=[record.mention_vector.copy()],
        visual_history=[record.visual_vector.copy()] if record.has_visual else [],
        text_count=1,
        visual_count=1 if record.has_visual else 0,
        visual_reliability_sum=1.0 if record.has_visual else 0.0,
        last_visual_reliability=1.0 if record.has_visual else 0.0,
        first_stream_index=record.stream_index,
        last_stream_index=record.stream_index,
        last_doc_id=record.doc_id,
        doc_ids={record.doc_id} if record.doc_id else set(),
        anchor_titles={anchor_title} if anchor_title else set(),
        context_titles=context_titles,
        anchor_token_set=anchor_token_set,
    )


def visual_reliability_score(slot: MemorySlot, record: StreamingMentionRecord) -> float:
    if not record.has_visual:
        return 0.0
    text_similarity = _cosine_similarity(slot.prototype_text, record.mention_vector)
    visual_similarity = (
        _cosine_similarity(slot.prototype_visual, record.visual_vector)
        if slot.visual_count > 0
        else 0.0
    )
    recent_visual_similarity = (
        _cosine_similarity(slot.visual_history[-1], record.visual_vector)
        if slot.visual_history
        else visual_similarity
    )
    normalized_text = 0.5 * (max(min(text_similarity, 1.0), -1.0) + 1.0)
    normalized_visual = 0.5 * (max(min(visual_similarity, 1.0), -1.0) + 1.0)
    normalized_recent_visual = 0.5 * (max(min(recent_visual_similarity, 1.0), -1.0) + 1.0)
    modality_agreement = max(text_similarity, 0.0) * max(visual_similarity, 0.0)
    visual_history_prior = min(slot.visual_count / max(len(slot.mentions), 1), 1.0)
    if slot.visual_count <= 0:
        normalized_visual = 0.55
        normalized_recent_visual = 0.55
        visual_history_prior = 0.25
    score = (
        0.25 * normalized_text
        + 0.30 * normalized_visual
        + 0.20 * normalized_recent_visual
        + 0.15 * modality_agreement
        + 0.10 * visual_history_prior
    )
    return float(max(0.0, min(score, 1.0)))


def update_memory_slot(
    slot: MemorySlot,
    record: StreamingMentionRecord,
    visual_update_mode: str = "always",
    visual_reliability_threshold: float = 0.0,
    visual_update_rate: float = 1.0,
) -> None:
    text_count = max(slot.text_count, len(slot.text_history), 1)
    slot.prototype_text = (
        slot.prototype_text * text_count + record.mention_vector
    ) / (text_count + 1)
    slot.text_count = text_count + 1
    visual_reliability = visual_reliability_score(slot, record)
    should_update_visual = bool(record.has_visual and visual_update_mode != "off")
    if visual_update_mode == "reliability_gated":
        should_update_visual = bool(should_update_visual and visual_reliability >= visual_reliability_threshold)
    if should_update_visual:
        visual_count = max(slot.visual_count, len(slot.visual_history), 0)
        if visual_update_mode == "reliability_gated" and visual_count > 0:
            alpha = max(0.0, min(1.0, float(visual_update_rate) * visual_reliability))
            slot.prototype_visual = (
                (1.0 - alpha) * slot.prototype_visual + alpha * record.visual_vector
            )
        else:
            slot.prototype_visual = (
                slot.prototype_visual * visual_count + record.visual_vector
            ) / (visual_count + 1)
        slot.visual_count = visual_count + 1
        slot.visual_history.append(record.visual_vector.copy())
        slot.visual_reliability_sum += visual_reliability
        slot.last_visual_reliability = visual_reliability
    elif record.has_visual:
        slot.last_visual_reliability = visual_reliability
    slot.mentions.append(record.mention_id)
    slot.text_history.append(record.mention_vector.copy())
    slot.last_stream_index = record.stream_index
    slot.last_doc_id = record.doc_id
    if record.doc_id:
        slot.doc_ids.add(record.doc_id)
    anchor_title = record_anchor_title(record)
    if anchor_title:
        slot.anchor_titles.add(anchor_title)
    slot.context_titles.update(record_context_titles(record))
    slot.anchor_token_set.update(record_anchor_token_set(record))


def memory_maturity_features(
    slot: MemorySlot,
    record: StreamingMentionRecord,
) -> dict[str, float]:
    slot_size = float(len(slot.mentions))
    unique_doc_count = float(len(slot.doc_ids))
    stream_age = float(max(record.stream_index - slot.last_stream_index, 0))
    stream_span = float(max(record.stream_index - slot.first_stream_index, 0))
    density = slot_size / max(stream_span + 1.0, 1.0)
    record_context = set(record_context_titles(record))
    record_anchor_tokens = record_anchor_token_set(record)
    anchor_context_overlap = anchor_overlap_jaccard(slot.context_titles, record_context)
    anchor_token_overlap = anchor_overlap_jaccard(slot.anchor_token_set, record_anchor_tokens)
    visual_coverage = float(slot.visual_count / max(slot_size, 1.0))
    visual_available = float(record.has_visual and slot.visual_count > 0)
    text_dispersion = float(
        np.std(
            [
                _cosine_similarity(vec, slot.prototype_text)
                for vec in slot.text_history
            ],
            dtype=np.float32,
        )
    )
    visual_dispersion = float(
        np.std(
            [
                _cosine_similarity(vec, slot.prototype_visual)
                for vec in slot.visual_history
            ],
            dtype=np.float32,
        )
    ) if slot.visual_history else 0.0
    text_similarity = _cosine_similarity(slot.prototype_text, record.mention_vector)
    visual_similarity = (
        _cosine_similarity(slot.prototype_visual, record.visual_vector)
        if record.has_visual and slot.visual_count > 0
        else 0.0
    )
    cross_modal_agreement = max(text_similarity, 0.0) * max(visual_similarity, 0.0)

    raw_maturity_features = np.array([
        math.log1p(slot_size),
        math.log1p(unique_doc_count),
        math.log1p(stream_span),
        density,
        anchor_token_overlap,
        anchor_context_overlap,
        visual_coverage,
        cross_modal_agreement,
        text_dispersion,
        visual_dispersion,
        math.log1p(stream_age),
    ], dtype=np.float32)

    return {
        "raw_maturity_features": raw_maturity_features,
        "slot_size": slot_size,
        "unique_doc_count": unique_doc_count,
        "stream_age": stream_age,
        "stream_span": stream_span,
        "density": float(density),
        "anchor_context_overlap": float(anchor_context_overlap),
        "anchor_token_overlap": float(anchor_token_overlap),
        "visual_coverage": float(visual_coverage),
        "visual_available": float(visual_available),
        "text_dispersion": float(text_dispersion),
        "visual_dispersion": float(visual_dispersion),
        "cross_modal_agreement": float(cross_modal_agreement),
    }


def memory_maturity_raw_feature_dim() -> int:
    return 11


class OnlinePrototypeMemory:
    def __init__(
        self,
        threshold: float = 0.72,
        margin_threshold: float = 0.0,
        text_weight: float = 1.0,
        visual_weight: float = 0.0,
        lemma_bonus: float = 0.0,
        visual_gate: float = -1.0,
        multiplicative_visual: bool = False,
        exemplar_text_weight: float = 0.0,
        exemplar_visual_weight: float = 0.0,
    ):
        self.threshold = threshold
        self.margin_threshold = margin_threshold
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        self.lemma_bonus = lemma_bonus
        self.visual_gate = visual_gate
        self.multiplicative_visual = multiplicative_visual
        self.exemplar_text_weight = exemplar_text_weight
        self.exemplar_visual_weight = exemplar_visual_weight
        self._next_id = 0
        self.memories: list[MemorySlot] = []

    def _allocate_memory(self, record: StreamingMentionRecord) -> MemorySlot:
        self._next_id += 1
        slot = new_memory_slot(f"mem_{self._next_id}", record)
        self.memories.append(slot)
        return slot

    def _score(self, slot: MemorySlot, record: StreamingMentionRecord) -> float:
        text_similarity = _cosine_similarity(slot.prototype_text, record.mention_vector)
        score = self.text_weight * text_similarity
        if self.exemplar_text_weight > 0.0 and slot.text_history:
            exemplar_text_similarity = max(
                _cosine_similarity(vec, record.mention_vector) for vec in slot.text_history
            )
            score += self.exemplar_text_weight * exemplar_text_similarity
        if self.visual_weight > 0.0 and record.has_visual:
            visual_similarity = _cosine_similarity(
                slot.prototype_visual, record.visual_vector
            )
            if text_similarity >= self.visual_gate:
                if self.multiplicative_visual:
                    score += (
                        self.visual_weight
                        * max(text_similarity, 0.0)
                        * max(visual_similarity, 0.0)
                    )
                else:
                    score += self.visual_weight * visual_similarity
        if self.exemplar_visual_weight > 0.0 and record.has_visual and slot.visual_history:
            exemplar_visual_similarity = max(
                _cosine_similarity(vec, record.visual_vector) for vec in slot.visual_history
            )
            if text_similarity >= self.visual_gate:
                score += self.exemplar_visual_weight * exemplar_visual_similarity
        if self.lemma_bonus > 0.0 and slot.prototype_lemma and slot.prototype_lemma == record.lemma:
            score += self.lemma_bonus
        return score

    def step(self, record: StreamingMentionRecord) -> EpisodePrediction:
        if not self.memories:
            slot = self._allocate_memory(record)
            return EpisodePrediction(
                mention_id=record.mention_id,
                gold_cluster=record.gold_cluster,
                predicted_memory_id=slot.memory_id,
                gold_action=record.gold_action,
                predicted_action="CREATE",
                score=1.0,
                episode_id=record.episode_id,
            )

        best_slot = None
        best_score = -math.inf
        second_best_score = -math.inf
        for slot in self.memories:
            score = self._score(slot, record)
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_slot = slot
            elif score > second_best_score:
                second_best_score = score

        margin = best_score - second_best_score if second_best_score > -math.inf else best_score
        if (
            best_slot is None
            or best_score < self.threshold
            or margin < self.margin_threshold
        ):
            slot = self._allocate_memory(record)
            return EpisodePrediction(
                mention_id=record.mention_id,
                gold_cluster=record.gold_cluster,
                predicted_memory_id=slot.memory_id,
                gold_action=record.gold_action,
                predicted_action="CREATE",
                score=margin if best_slot is not None else 0.0,
                episode_id=record.episode_id,
            )

        update_memory_slot(best_slot, record)
        return EpisodePrediction(
            mention_id=record.mention_id,
            gold_cluster=record.gold_cluster,
            predicted_memory_id=best_slot.memory_id,
            gold_action=record.gold_action,
            predicted_action="LINK",
            score=margin,
            episode_id=record.episode_id,
        )


def _prediction_cluster_id(prediction: EpisodePrediction) -> str:
    if prediction.episode_id:
        return f"{prediction.episode_id}::{prediction.predicted_memory_id}"
    return prediction.predicted_memory_id


def evaluate_episode_predictions(
    predictions: Iterable[EpisodePrediction],
) -> dict[str, float]:
    predictions = list(predictions)
    if not predictions:
        return {
            "num_mentions": 0,
            "action_accuracy": 0.0,
            "pair_precision": 0.0,
            "pair_recall": 0.0,
            "pair_f1": 0.0,
        }

    action_correct = sum(
        int(pred.gold_action == pred.predicted_action) for pred in predictions
    )
    tp = fp = fn = 0
    grouped: dict[str, list[EpisodePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.episode_id or "__global__"].append(prediction)
    for episode_predictions in grouped.values():
        for i in range(len(episode_predictions)):
            left_cluster_id = _prediction_cluster_id(episode_predictions[i])
            for j in range(i + 1, len(episode_predictions)):
                gold_same = episode_predictions[i].gold_cluster == episode_predictions[j].gold_cluster
                pred_same = left_cluster_id == _prediction_cluster_id(episode_predictions[j])
                if gold_same and pred_same:
                    tp += 1
                elif pred_same and not gold_same:
                    fp += 1
                elif gold_same and not pred_same:
                    fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "num_mentions": len(predictions),
        "action_accuracy": action_correct / len(predictions),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
    }


def memory_candidate_features(
    slot: MemorySlot,
    record: StreamingMentionRecord,
) -> np.ndarray:
    text_similarity = _cosine_similarity(slot.prototype_text, record.mention_vector)
    visual_similarity = (
        _cosine_similarity(slot.prototype_visual, record.visual_vector)
        if record.has_visual and slot.visual_count > 0
        else 0.0
    )
    recent_text_similarity = _cosine_similarity(slot.text_history[-1], record.mention_vector)
    recent_visual_similarity = (
        _cosine_similarity(slot.visual_history[-1], record.visual_vector)
        if record.has_visual and slot.visual_history
        else 0.0
    )
    exemplar_text_similarity = max(
        (_cosine_similarity(vec, record.mention_vector) for vec in slot.text_history),
        default=0.0,
    )
    exemplar_visual_similarity = max(
        (_cosine_similarity(vec, record.visual_vector) for vec in slot.visual_history),
        default=0.0,
    )
    same_lemma = float(bool(slot.prototype_lemma and slot.prototype_lemma == record.lemma))
    same_doc = float(
        any(record.doc_id == mention_id.rsplit("_", 1)[0] for mention_id in slot.mentions)
    )
    same_recent_doc = float(bool(slot.last_doc_id and slot.last_doc_id == record.doc_id))
    slot_size = float(len(slot.mentions))
    unique_doc_count = float(len(slot.doc_ids))
    stream_age = float(max(record.stream_index - slot.last_stream_index, 0))
    stream_span = float(max(record.stream_index - slot.first_stream_index, 0))
    density = slot_size / max(stream_span + 1.0, 1.0)
    interaction = max(text_similarity, 0.0) * max(visual_similarity, 0.0)
    text_spread = exemplar_text_similarity - text_similarity
    visual_spread = exemplar_visual_similarity - visual_similarity
    modality_gap = abs(text_similarity - visual_similarity)
    record_title = record_anchor_title(record)
    record_context = set(record_context_titles(record))
    record_anchor_tokens = record_anchor_token_set(record)
    anchor_title_match = float(bool(record_title and record_title in slot.anchor_titles))
    anchor_context_overlap = anchor_overlap_jaccard(slot.context_titles, record_context)
    anchor_token_overlap = anchor_overlap_jaccard(slot.anchor_token_set, record_anchor_tokens)
    text_dispersion = float(
        np.std(
            [
                _cosine_similarity(vec, slot.prototype_text)
                for vec in slot.text_history
            ],
            dtype=np.float32,
        )
    )
    visual_dispersion = float(
        np.std(
            [
                _cosine_similarity(vec, slot.prototype_visual)
                for vec in slot.visual_history
            ],
            dtype=np.float32,
        )
    ) if slot.visual_history else 0.0
    visual_coverage = float(slot.visual_count / max(slot_size, 1.0))
    cross_modal_agreement = max(text_similarity, 0.0) * max(visual_similarity, 0.0)

    return np.asarray(
        [
            text_similarity,
            visual_similarity,
            exemplar_text_similarity,
            exemplar_visual_similarity,
            recent_text_similarity,
            recent_visual_similarity,
            same_lemma,
            same_doc,
            same_recent_doc,
            math.log1p(slot_size),
            math.log1p(unique_doc_count),
            math.log1p(stream_age),
            math.log1p(stream_span),
            density,
            text_spread,
            visual_spread,
            modality_gap,
            anchor_title_match,
            anchor_context_overlap,
            anchor_token_overlap,
            text_dispersion,
            visual_dispersion,
            interaction,
            float(record.has_visual),
            math.log1p(slot_size),
            math.log1p(unique_doc_count),
            math.log1p(stream_span),
            density,
            anchor_token_overlap,
            anchor_context_overlap,
            visual_coverage,
            cross_modal_agreement,
            text_dispersion,
            visual_dispersion,
            math.log1p(stream_age),
        ],
        dtype=np.float32,
    )


def memory_candidate_feature_dim() -> int:
    return 35


def memory_candidate_feature_names() -> list[str]:
    return [
        "text_similarity",
        "visual_similarity",
        "exemplar_text_similarity",
        "exemplar_visual_similarity",
        "recent_text_similarity",
        "recent_visual_similarity",
        "same_lemma",
        "same_doc",
        "same_recent_doc",
        "slot_size_log",
        "unique_doc_count_log",
        "stream_age_log",
        "stream_span_log",
        "density",
        "text_spread",
        "visual_spread",
        "modality_gap",
        "anchor_title_match",
        "anchor_context_overlap",
        "anchor_token_overlap",
        "text_dispersion",
        "visual_dispersion",
        "interaction",
        "has_visual",
        "maturity_slot_size_log",
        "maturity_unique_doc_count_log",
        "maturity_stream_span_log",
        "maturity_density",
        "maturity_anchor_token_overlap",
        "maturity_anchor_context_overlap",
        "maturity_visual_coverage",
        "maturity_cross_modal_agreement",
        "maturity_text_dispersion",
        "maturity_visual_dispersion",
        "maturity_stream_age_log",
    ]


def memory_candidate_feature_group_indices() -> dict[str, list[int]]:
    return {
        "maturity": [9, 10, 11, 12, 13, 20, 21],
        "maturity_scoring": [11, 12, 20, 21],
        "state": [9, 10, 11, 12, 13],
        "visual": [1, 3, 5, 15, 16, 21, 22, 23],
        "anchor_history": [17, 18, 19],
        "learnable_maturity": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
    }


def shortlist_candidate_row_dim() -> int:
    return memory_candidate_feature_dim() + 5


def shortlist_aggregate_feature_dim() -> int:
    return memory_candidate_feature_dim() * 2 + 6


def save_records(records: Iterable[StreamingMentionRecord], output_path: Path) -> None:
    import pickle

    serializable = [record.to_storage_dict() for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(serializable, f)


def load_records(input_path: Path) -> list[StreamingMentionRecord]:
    import pickle

    with open(input_path, "rb") as f:
        payload = pickle.load(f)
    records: list[StreamingMentionRecord] = []
    for item in payload:
        text_dim = _infer_dim(item.get("mention_vector"), 300)
        visual_dim = _infer_dim(item.get("visual_vector"), 512)
        records.append(
            StreamingMentionRecord(
                mention_id=item["mention_id"],
                split=item["split"],
                episode_id=item["episode_id"],
                stream_index=int(item["stream_index"]),
                doc_id=item["doc_id"],
                sentence_id=int(item["sentence_id"]),
                mention_text=item["mention_text"],
                bert_doc=item["bert_doc"],
                topic=item["topic"],
                predicted_topic=item["predicted_topic"],
                gold_cluster=item["gold_cluster"],
                gold_action=item["gold_action"],
                has_visual=bool(item["has_visual"]),
                mention_vector=_to_numpy(item["mention_vector"], text_dim),
                visual_vector=_to_numpy(item["visual_vector"], visual_dim),
                lemma=item["lemma"],
            )
        )
    return records


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
