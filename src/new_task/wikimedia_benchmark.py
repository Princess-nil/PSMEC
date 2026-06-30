from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from requests import Response

USER_AGENT = "MCDECR-SMEMC-BenchmarkBot/0.1 (research use; local workspace runner)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

SECTION_RE = re.compile(r"^'''(.+?)'''$")
BULLET_RE = re.compile(r"^(\*+)\s*(.+?)\s*$")
WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
EXTERNAL_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

EVENT_KEYWORDS = {
    "accident",
    "attack",
    "battle",
    "blast",
    "bombing",
    "campaign",
    "ceasefire",
    "collision",
    "conflict",
    "crash",
    "crisis",
    "cyclone",
    "demonstration",
    "disaster",
    "earthquake",
    "election",
    "epidemic",
    "eruption",
    "evacuation",
    "explosion",
    "flood",
    "hurricane",
    "incident",
    "invasion",
    "landslide",
    "massacre",
    "mission",
    "operation",
    "outbreak",
    "protest",
    "referendum",
    "riot",
    "shooting",
    "spill",
    "storm",
    "strike",
    "summit",
    "talks",
    "test",
    "tornado",
    "train derailment",
    "trial",
    "tsunami",
    "war",
    "wildfire",
}

NON_EVENT_HINTS = {
    "biography",
    "bilateral relations",
    "capital and largest city",
    "company",
    "country",
    "county",
    "district",
    "footballer",
    "foreign relations",
    "list of",
    "organization",
    "person",
    "politician",
    "province",
    "relations",
    "revenue service",
    "river",
    "state of",
    "town",
    "unrecognized state",
    "village",
}


@dataclass
class WikiLink:
    target: str
    label: str


@dataclass
class ExternalReference:
    url: str
    label: str


@dataclass
class BulletNode:
    node_id: str
    event_date: str
    order: int
    level: int
    section: str
    raw_text: str
    plain_text: str
    internal_links: list[WikiLink]
    external_refs: list[ExternalReference]
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)


@dataclass
class PageMetadata:
    title: str
    normalized_title: str
    pageid: int | None
    wikidata_qid: str
    description: str
    extract: str
    categories: list[str]
    pageimage: str
    thumbnail_url: str
    original_url: str


@dataclass
class WikimediaMentionRecord:
    mention_id: str
    split: str
    event_date: str
    episode_id: str
    stream_index: int
    doc_id: str
    section: str
    mention_text: str
    context_path: list[str]
    linked_titles: list[str]
    source_refs: list[dict[str, str]]
    primary_event_title: str
    primary_event_qid: str
    primary_event_description: str
    primary_event_extract: str
    primary_event_categories: list[str]
    gold_cluster: str
    gold_action: str
    image_path: str
    image_url: str
    image_name: str
    thumbnail_url: str
    has_visual: bool

    def to_dict(self) -> dict:
        return asdict(self)


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def normalize_title(title: str) -> str:
    return title.replace("_", " ").strip()


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = NON_WORD_RE.sub("-", text)
    return text.strip("-") or "unknown"


def extract_internal_links(text: str) -> list[WikiLink]:
    links: list[WikiLink] = []
    for match in WIKI_LINK_RE.finditer(text):
        target = normalize_title(match.group(1))
        label = normalize_title(match.group(2) or match.group(1))
        if target.startswith(("Category:", "File:", "Template:", "Portal:")):
            continue
        links.append(WikiLink(target=target, label=label))
    return links


def extract_external_refs(text: str) -> list[ExternalReference]:
    refs: list[ExternalReference] = []
    for match in EXTERNAL_LINK_RE.finditer(text):
        refs.append(ExternalReference(url=match.group(1), label=match.group(2).strip()))
    return refs


def render_plain_text(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = EXTERNAL_LINK_RE.sub(lambda match: match.group(2).strip(), text)
    text = WIKI_LINK_RE.sub(lambda match: normalize_title(match.group(2) or match.group(1)), text)
    text = re.sub(r"''+", "", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_current_events_template(wikitext: str) -> str:
    lines = []
    capture = False
    for raw_line in wikitext.splitlines():
        line = raw_line.rstrip()
        if line.startswith("{{Current events"):
            capture = True
            continue
        if capture and line == "}}":
            break
        if capture and line.startswith("|content="):
            line = line[len("|content=") :]
        if capture:
            lines.append(line)
    return "\n".join(lines)


class WikimediaApiClient:
    def __init__(
        self,
        sleep_seconds: float = 0.1,
        timeout: int = 30,
        max_retries: int = 6,
        metadata_batch_size: int = 10,
        retry_forever: bool = True,
        max_retry_sleep_seconds: float = 120.0,
        max_image_retries: int = 12,
        image_rate_limit_cooldown_seconds: float = 30.0,
        image_rate_limit_max_cooldown_seconds: float = 600.0,
    ):
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.metadata_batch_size = metadata_batch_size
        self.retry_forever = retry_forever
        self.max_retry_sleep_seconds = max_retry_sleep_seconds
        self.max_image_retries = max_image_retries
        self.image_rate_limit_cooldown_seconds = image_rate_limit_cooldown_seconds
        self.image_rate_limit_max_cooldown_seconds = image_rate_limit_max_cooldown_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._host_retry_not_before: dict[str, float] = {}
        self._host_consecutive_429: dict[str, int] = {}

    @staticmethod
    def _failed_marker_path(output_path: Path) -> Path:
        suffix = f"{output_path.suffix}.failed" if output_path.suffix else ".failed"
        return output_path.with_suffix(suffix)

    def _retry_delay_seconds(self, response: Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after:
                try:
                    delay = max(float(retry_after), self.sleep_seconds, 1.0)
                    return min(delay, self.max_retry_sleep_seconds)
                except ValueError:
                    try:
                        retry_dt = parsedate_to_datetime(retry_after)
                        delay = retry_dt.timestamp() - time.time()
                        delay = max(delay, self.sleep_seconds, 1.0)
                        return min(delay, self.max_retry_sleep_seconds)
                    except (TypeError, ValueError, OverflowError):
                        pass
        return min(max(self.sleep_seconds * (2**attempt), 1.0), self.max_retry_sleep_seconds)

    def _wait_for_host_cooldown(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        if not host:
            return
        wait_until = self._host_retry_not_before.get(host, 0.0)
        remaining = wait_until - time.time()
        if remaining <= 0.0:
            return
        print(
            {
                "stage": "image_host_cooldown_wait",
                "host": host,
                "delay_seconds": round(remaining, 2),
            },
            flush=True,
        )
        time.sleep(remaining)

    def _register_image_rate_limit(self, url: str, response: Response | None, attempt: int) -> tuple[float, int]:
        host = urlparse(url).netloc.lower()
        base_delay = self._retry_delay_seconds(response, attempt)
        consecutive_429 = self._host_consecutive_429.get(host, 0) + 1 if host else 1
        if host:
            self._host_consecutive_429[host] = consecutive_429
        cooldown_delay = min(
            self.image_rate_limit_cooldown_seconds * (2 ** min(consecutive_429 - 1, 4)),
            self.image_rate_limit_max_cooldown_seconds,
        )
        delay = max(base_delay, cooldown_delay)
        if host:
            self._host_retry_not_before[host] = max(
                self._host_retry_not_before.get(host, 0.0),
                time.time() + delay,
            )
        return delay, consecutive_429

    def _clear_image_rate_limit(self, url: str) -> None:
        host = urlparse(url).netloc.lower()
        if not host:
            return
        self._host_consecutive_429.pop(host, None)
        self._host_retry_not_before.pop(host, None)

    def _get_json(self, url: str, params: dict, max_attempts: int | None = None) -> dict:
        response: Response | None = None
        attempt = 0
        while True:
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if max_attempts is not None and attempt + 1 >= max_attempts:
                        raise RuntimeError(
                            f"request exceeded max_attempts={max_attempts} "
                            f"status_code={response.status_code}"
                        )
                    delay = self._retry_delay_seconds(response, attempt)
                    print(
                        {
                            "stage": "request_retry",
                            "status_code": response.status_code,
                            "delay_seconds": round(delay, 2),
                            "attempt": attempt + 1,
                            "params": {key: params.get(key) for key in ("action", "page", "titles", "prop")},
                        },
                        flush=True,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                response.raise_for_status()
                payload = response.json()
                if self.sleep_seconds > 0.0:
                    time.sleep(self.sleep_seconds)
                return payload
            except requests.RequestException:
                if max_attempts is not None and attempt + 1 >= max_attempts:
                    raise
                if not self.retry_forever and attempt + 1 >= self.max_retries:
                    raise
                delay = self._retry_delay_seconds(response, attempt)
                print(
                    {
                        "stage": "request_exception_retry",
                        "delay_seconds": round(delay, 2),
                        "attempt": attempt + 1,
                        "params": {key: params.get(key) for key in ("action", "page", "titles", "prop")},
                    },
                    flush=True,
                )
                time.sleep(delay)
                attempt += 1

    def fetch_current_events_wikitext(self, day: date) -> str:
        page = f"Portal:Current_events/{day.strftime('%Y_%B_%-d')}"
        params = {
            "action": "parse",
            "page": page,
            "prop": "wikitext",
            "format": "json",
            "formatversion": 2,
        }
        payload = self._get_json(WIKIPEDIA_API, params)
        return str(payload["parse"]["wikitext"])

    def fetch_page_metadata(self, titles: list[str]) -> dict[str, PageMetadata]:
        metadata: dict[str, PageMetadata] = {}
        normalized_titles = sorted({normalize_title(title) for title in titles if title})

        def fetch_batch(batch: list[str]) -> None:
            if not batch:
                return
            params = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "titles": "|".join(batch),
                "prop": "pageprops|pageimages|extracts|description|categories",
                "ppprop": "wikibase_item",
                "piprop": "name|thumbnail|original",
                "pithumbsize": 512,
                "exintro": 1,
                "explaintext": 1,
                "exchars": 320,
                "cllimit": 10,
            }
            try:
                payload = self._get_json(
                    WIKIPEDIA_API,
                    params,
                    max_attempts=min(self.max_retries, 4) if self.retry_forever else self.max_retries,
                )
            except Exception as exc:
                if len(batch) > 1:
                    split = max(1, len(batch) // 2)
                    left = batch[:split]
                    right = batch[split:]
                    print(
                        {
                            "stage": "metadata_batch_split",
                            "batch_size": len(batch),
                            "left_size": len(left),
                            "right_size": len(right),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "titles": batch,
                        },
                        flush=True,
                    )
                    fetch_batch(left)
                    fetch_batch(right)
                    return
                print(
                    {
                        "stage": "metadata_single_skip",
                        "title": batch[0],
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                    flush=True,
                )
                return
            pages = payload.get("query", {}).get("pages", [])
            for page in pages:
                title = normalize_title(page.get("title", ""))
                metadata[title] = PageMetadata(
                    title=title,
                    normalized_title=title,
                    pageid=page.get("pageid"),
                    wikidata_qid=str(page.get("pageprops", {}).get("wikibase_item", "")),
                    description=str(page.get("description", "")),
                    extract=str(page.get("extract", "")),
                    categories=[
                        str(item.get("title", "")).replace("Category:", "")
                        for item in page.get("categories", [])
                    ],
                    pageimage=str(page.get("pageimage", "")),
                    thumbnail_url=str(page.get("thumbnail", {}).get("source", "")),
                    original_url=str(page.get("original", {}).get("source", "")),
                )
        for start in range(0, len(normalized_titles), self.metadata_batch_size):
            batch = normalized_titles[start : start + self.metadata_batch_size]
            fetch_batch(batch)
        return metadata

    def download_file(self, url: str, output_path: Path) -> bool:
        if not url:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            return True
        failed_marker = self._failed_marker_path(output_path)
        if failed_marker.exists():
            return False
        response: Response | None = None
        attempt = 0
        while True:
            try:
                self._wait_for_host_cooldown(url)
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt + 1 >= self.max_image_retries:
                        failed_marker.write_text(
                            json.dumps(
                                {
                                    "status": "failed",
                                    "status_code": response.status_code,
                                    "attempts": attempt + 1,
                                    "url": url,
                                },
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )
                        print(
                            {
                                "stage": "image_skip_after_retries",
                                "status_code": response.status_code,
                                "attempts": attempt + 1,
                                "url": url,
                            },
                            flush=True,
                        )
                        return False
                    if response.status_code == 429:
                        delay, consecutive_429 = self._register_image_rate_limit(url, response, attempt)
                    else:
                        delay = self._retry_delay_seconds(response, attempt)
                        consecutive_429 = self._host_consecutive_429.get(urlparse(url).netloc.lower(), 0)
                    print(
                        {
                            "stage": "image_retry",
                            "status_code": response.status_code,
                            "delay_seconds": round(delay, 2),
                            "attempt": attempt + 1,
                            "consecutive_host_429": consecutive_429,
                            "url": url,
                        },
                        flush=True,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                response.raise_for_status()
                output_path.write_bytes(response.content)
                self._clear_image_rate_limit(url)
                if failed_marker.exists():
                    failed_marker.unlink()
                if self.sleep_seconds > 0.0:
                    time.sleep(self.sleep_seconds)
                return True
            except requests.RequestException:
                if attempt + 1 >= self.max_image_retries:
                    failed_marker.write_text(
                        json.dumps(
                            {
                                "status": "exception_failed",
                                "attempts": attempt + 1,
                                "url": url,
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        {
                            "stage": "image_exception_skip_after_retries",
                            "attempts": attempt + 1,
                            "url": url,
                        },
                        flush=True,
                    )
                    return False
                if not self.retry_forever and attempt + 1 >= self.max_retries:
                    raise
                delay = self._retry_delay_seconds(response, attempt)
                print(
                    {
                        "stage": "image_exception_retry",
                        "delay_seconds": round(delay, 2),
                        "attempt": attempt + 1,
                        "url": url,
                    },
                    flush=True,
                )
                time.sleep(delay)
                attempt += 1


def parse_current_events_page(event_date: str, wikitext: str) -> list[BulletNode]:
    content = strip_current_events_template(wikitext)
    nodes: list[BulletNode] = []
    current_section = "unknown"
    stack: dict[int, BulletNode] = {}
    order = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<!--"):
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = render_plain_text(section_match.group(1))
            stack.clear()
            continue
        bullet_match = BULLET_RE.match(line)
        if not bullet_match:
            continue

        level = len(bullet_match.group(1))
        raw_text = bullet_match.group(2)
        node = BulletNode(
            node_id=f"{event_date}::{order}",
            event_date=event_date,
            order=order,
            level=level,
            section=current_section,
            raw_text=raw_text,
            plain_text=render_plain_text(raw_text),
            internal_links=extract_internal_links(raw_text),
            external_refs=extract_external_refs(raw_text),
            parent_id=stack[level - 1].node_id if level > 1 and (level - 1) in stack else None,
        )
        if node.parent_id is not None:
            stack[level - 1].child_ids.append(node.node_id)
        stack[level] = node
        for key in list(stack.keys()):
            if key > level:
                del stack[key]
        nodes.append(node)
        order += 1

    return nodes


def build_parent_index(nodes: list[BulletNode]) -> dict[str, BulletNode]:
    return {node.node_id: node for node in nodes}


def get_context_path(node: BulletNode, node_index: dict[str, BulletNode]) -> list[str]:
    path: list[str] = []
    current = node
    while current.parent_id is not None:
        current = node_index[current.parent_id]
        anchor = current.internal_links[0].target if current.internal_links else current.plain_text
        path.append(anchor)
    path.reverse()
    return path


def collect_candidate_titles(
    node: BulletNode,
    node_index: dict[str, BulletNode],
) -> list[tuple[str, int, int, bool]]:
    candidates: list[tuple[str, int, int, bool]] = []
    current: BulletNode | None = node
    ancestor_distance = 0
    while current is not None:
        for order, link in enumerate(current.internal_links):
            is_anchor = bool(order == 0 and (ancestor_distance > 0 or current.level == 1 or current.child_ids))
            candidates.append((link.target, ancestor_distance, order, is_anchor))
        current = node_index.get(current.parent_id or "")
        ancestor_distance += 1
    return candidates


def score_candidate_title(
    title: str,
    metadata: PageMetadata | None,
    ancestor_distance: int,
    link_order: int,
    is_anchor: bool,
) -> float:
    text = normalize_title(title).lower()
    description = (metadata.description if metadata else "").lower()
    extract = (metadata.extract if metadata else "").lower()
    categories = " ".join(metadata.categories if metadata else []).lower()

    score = 0.0
    if YEAR_RE.search(text):
        score += 0.6
    if ancestor_distance == 0:
        score += 1.1
    elif ancestor_distance == 1:
        score += 1.4
    else:
        score += max(0.8 - 0.05 * ancestor_distance, 0.0)
    if is_anchor:
        score += 1.2
    score += max(0.0, 0.2 - 0.07 * link_order)

    for keyword in EVENT_KEYWORDS:
        if keyword in text:
            score += 1.2
        if keyword in description:
            score += 1.0
        if keyword in extract:
            score += 0.5
        if keyword in categories:
            score += 0.5

    for hint in NON_EVENT_HINTS:
        if hint in text:
            score -= 0.8
        if hint in description:
            score -= 0.6
        if hint in categories:
            score -= 0.4
    if len(text.split()) <= 3 and not YEAR_RE.search(text):
        score -= 0.4
    if not any(keyword in text for keyword in EVENT_KEYWORDS) and not YEAR_RE.search(text):
        score -= 0.3
    if metadata is None:
        score -= 0.2
    return score


def choose_primary_event_title(
    node: BulletNode,
    node_index: dict[str, BulletNode],
    metadata_map: dict[str, PageMetadata],
) -> tuple[str, float]:
    candidates = collect_candidate_titles(node, node_index)
    if not candidates:
        return node.plain_text, -math.inf

    scored: list[tuple[float, str]] = []
    for title, ancestor_distance, link_order, is_anchor in candidates:
        metadata = metadata_map.get(normalize_title(title))
        score = score_candidate_title(
            title=title,
            metadata=metadata,
            ancestor_distance=ancestor_distance,
            link_order=link_order,
            is_anchor=is_anchor,
        )
        scored.append((score, normalize_title(title)))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_title = scored[0]
    if best_score > 0.0:
        return best_title, best_score

    for title, ancestor_distance, _, _ in candidates:
        if ancestor_distance > 0:
            return normalize_title(title), best_score
    return normalize_title(candidates[0][0]), best_score


def derive_leaf_mentions(nodes: list[BulletNode]) -> list[BulletNode]:
    leaves: list[BulletNode] = []
    for node in nodes:
        if node.child_ids:
            continue
        if len(node.plain_text) < 24:
            continue
        leaves.append(node)
    return leaves


def choose_image_url(metadata: PageMetadata) -> str:
    if metadata.thumbnail_url:
        return metadata.thumbnail_url
    return metadata.original_url


def infer_image_suffix(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".svg", ".webp", ".gif"}:
        return suffix
    return ".jpg"


def is_event_like_primary(title: str, metadata: PageMetadata | None) -> bool:
    text = normalize_title(title).lower()
    description = (metadata.description if metadata else "").lower()
    categories = " ".join(metadata.categories if metadata else []).lower()
    if YEAR_RE.search(text):
        return True
    if any(hint in text for hint in NON_EVENT_HINTS) or any(hint in description for hint in NON_EVENT_HINTS):
        return False
    haystacks = [text, description, categories]
    if any(keyword in haystack for haystack in haystacks for keyword in EVENT_KEYWORDS):
        return True
    return False


def assign_temporal_splits(
    event_dates: list[str],
    train_ratio: float,
    dev_ratio: float,
) -> dict[str, str]:
    unique_dates = sorted(set(event_dates))
    if not unique_dates:
        return {}
    train_end = max(1, int(len(unique_dates) * train_ratio))
    dev_end = max(train_end + 1, int(len(unique_dates) * (train_ratio + dev_ratio)))
    if len(unique_dates) >= 3:
        train_end = min(train_end, len(unique_dates) - 2)
        dev_end = min(max(dev_end, train_end + 1), len(unique_dates) - 1)
    split_map: dict[str, str] = {}
    for index, item in enumerate(unique_dates):
        if index < train_end:
            split_map[item] = "train"
        elif index < dev_end:
            split_map[item] = "dev"
        else:
            split_map[item] = "test"
    return split_map


def build_wikimedia_mentions(
    nodes_by_day: dict[str, list[BulletNode]],
    metadata_map: dict[str, PageMetadata],
    split_map: dict[str, str],
    image_dir: Path | None,
    api_client: WikimediaApiClient | None = None,
    min_primary_score: float = 1.4,
) -> list[WikimediaMentionRecord]:
    records: list[WikimediaMentionRecord] = []
    episode_seen_clusters: dict[str, set[str]] = {}
    episode_stream_index: dict[str, int] = {}

    for event_date in sorted(nodes_by_day):
        nodes = nodes_by_day[event_date]
        node_index = build_parent_index(nodes)
        leaves = derive_leaf_mentions(nodes)
        for leaf in leaves:
            primary_event_title, primary_score = choose_primary_event_title(
                node=leaf,
                node_index=node_index,
                metadata_map=metadata_map,
            )
            if primary_score < min_primary_score:
                continue
            primary_metadata = metadata_map.get(
                primary_event_title,
                PageMetadata(
                    title=primary_event_title,
                    normalized_title=primary_event_title,
                    pageid=None,
                    wikidata_qid="",
                    description="",
                    extract="",
                    categories=[],
                    pageimage="",
                    thumbnail_url="",
                    original_url="",
                ),
            )
            if not is_event_like_primary(primary_event_title, primary_metadata):
                continue
            if (
                not primary_metadata.wikidata_qid
                and not YEAR_RE.search(primary_event_title)
                and not get_context_path(leaf, node_index)
            ):
                continue
            qid = primary_metadata.wikidata_qid
            gold_cluster = qid if qid else slugify(primary_event_title)
            section_slug = slugify(leaf.section)
            month_key = event_date[:7]
            episode_id = f"{month_key}::{section_slug}"
            stream_index = episode_stream_index.get(episode_id, 0)
            episode_stream_index[episode_id] = stream_index + 1

            seen = episode_seen_clusters.setdefault(episode_id, set())
            gold_action = "CREATE" if gold_cluster not in seen else "LINK"
            seen.add(gold_cluster)

            image_url = choose_image_url(primary_metadata)
            image_path = ""
            if image_dir is not None and image_url and api_client is not None:
                suffix = infer_image_suffix(image_url)
                image_file = image_dir / f"{slugify(primary_event_title)}{suffix}"
                try:
                    if api_client.download_file(image_url, image_file):
                        image_path = str(image_file)
                except requests.RequestException:
                    image_path = ""

            records.append(
                WikimediaMentionRecord(
                    mention_id=leaf.node_id,
                    split=split_map.get(event_date, "train"),
                    event_date=event_date,
                    episode_id=episode_id,
                    stream_index=stream_index,
                    doc_id=event_date,
                    section=leaf.section,
                    mention_text=leaf.plain_text,
                    context_path=get_context_path(leaf, node_index),
                    linked_titles=[link.target for link in leaf.internal_links],
                    source_refs=[asdict(item) for item in leaf.external_refs],
                    primary_event_title=primary_event_title,
                    primary_event_qid=primary_metadata.wikidata_qid,
                    primary_event_description=primary_metadata.description,
                    primary_event_extract=primary_metadata.extract,
                    primary_event_categories=primary_metadata.categories,
                    gold_cluster=gold_cluster,
                    gold_action=gold_action,
                    image_path=image_path,
                    image_url=primary_metadata.original_url,
                    image_name=primary_metadata.pageimage,
                    thumbnail_url=primary_metadata.thumbnail_url,
                    has_visual=bool(image_path or primary_metadata.thumbnail_url or primary_metadata.original_url),
                )
            )
    return records


def build_summary(records: list[WikimediaMentionRecord], crawl_config: dict) -> dict:
    summary: dict[str, dict] = {"crawl_config": crawl_config}
    for split in ("train", "dev", "test"):
        split_records = [record for record in records if record.split == split]
        episode_ids = {record.episode_id for record in split_records}
        gold_clusters = {record.gold_cluster for record in split_records}
        summary[split] = {
            "num_mentions": len(split_records),
            "num_episodes": len(episode_ids),
            "num_clusters": len(gold_clusters),
            "num_visual_mentions": sum(int(record.has_visual) for record in split_records),
        }
    summary["overall"] = {
        "num_mentions": len(records),
        "num_days": len({record.event_date for record in records}),
        "num_sections": len({record.section for record in records}),
        "num_clusters": len({record.gold_cluster for record in records}),
    }
    return summary


def save_jsonl(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def collect_wikimedia_current_events(
    start_date: str,
    end_date: str,
    sleep_seconds: float = 0.1,
    max_retries: int = 6,
    metadata_batch_size: int = 10,
    retry_forever: bool = True,
    max_retry_sleep_seconds: float = 120.0,
) -> tuple[dict[str, list[BulletNode]], dict[str, PageMetadata]]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    api_client = WikimediaApiClient(
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        metadata_batch_size=metadata_batch_size,
        retry_forever=retry_forever,
        max_retry_sleep_seconds=max_retry_sleep_seconds,
    )

    nodes_by_day: dict[str, list[BulletNode]] = {}
    all_titles: list[str] = []
    for day in daterange(start, end):
        day_str = day.isoformat()
        try:
            wikitext = api_client.fetch_current_events_wikitext(day)
        except requests.RequestException:
            continue
        nodes = parse_current_events_page(day_str, wikitext)
        if not nodes:
            continue
        nodes_by_day[day_str] = nodes
        for node in nodes:
            all_titles.extend(link.target for link in node.internal_links)

    metadata_map = api_client.fetch_page_metadata(all_titles)
    return nodes_by_day, metadata_map
