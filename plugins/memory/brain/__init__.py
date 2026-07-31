"""hermes-brain — graph memory provider backed by the `brain` SPO knowledge graph.

Unlike vector stores (mem0), this provider keeps **typed relations**:
``subject —predicate→ object``. That lets the agent traverse links between facts
that share no semantic similarity (``Paul —rules→ Arrakis —contains→ spice``).

Why a MemoryProvider and not an MCP server: MCP tools are opt-in — the model has
to decide to call them, and in practice it does not (nanobot's brains stayed
empty for months). The MemoryProvider hooks are mandatory:

* ``prefetch``   — graph recall is injected into every turn's context
* ``sync_turn``  — every turn is distilled into triples and written back
* tools          — explicit traversal (neighbors / path / communities)

Backend: /home/dietpi/clawd/brain/cli.py (Postgres + pgvector, per-bot database).

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-brain:
      brain_db: paul_brain              # per-bot graph database
      cli_dir: /home/dietpi/clawd/brain # where cli.py + .venv live
      auto_store: true                  # distil each turn into triples
      extract_model: mistral/mistral-large-latest
      recall_limit: 8
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider


# --- Проверка происхождения точных величин ------------------------------------
# Модель читает числа с фотографий и документов и ошибается: один и тот же чек она
# подписывала тремя разными магазинами и валютами. Такая ошибка молча переприписывает
# реальные суммы чужому месту и времени, и опровергнуть её потом нечем.
#
# Принцип: точная величина или идентификатор — цена, вес, номер документа, индекс —
# попадает в долговременную память, только если её написал сам пользователь. Мелкие
# счётные числа ("3 яйца") под правило не подпадают: это не заявка на идентичность,
# и их модель берёт из смысла фразы, а не с картинки.

_PRECISE_TOKEN = re.compile(r"\d+(?:[.,]\d+)?(?:[/-]\d+)*")


def _is_precise(token: str) -> bool:
    """Точная величина: с дробной частью, составная (23/1157/188) или из 3+ цифр."""
    digits = re.sub(r"\D", "", token)
    return ("." in token or "," in token or "/" in token or "-" in token
            or len(digits) >= 3)


def _digit_key(token: str) -> str:
    return re.sub(r"\D", "", token)


def unsupported_precise_values(triple: Dict[str, str], user_text: str) -> List[str]:
    """Точные величины из тройки, которых нет в тексте пользователя."""
    claimed = " ".join(str(triple.get(k, "")) for k in ("subject", "predicate", "object", "fact"))
    known = {_digit_key(m) for m in _PRECISE_TOKEN.findall(user_text or "")}
    missing = []
    for token in _PRECISE_TOKEN.findall(claimed):
        if not _is_precise(token):
            continue
        key = _digit_key(token)
        if key and key not in known and not any(key in k for k in known):
            missing.append(token)
    return missing

logger = logging.getLogger(__name__)

_DEFAULT_CLI_DIR = "/home/dietpi/clawd/brain"
_DEFAULT_DB = "paul_brain"
_DEFAULT_EXTRACT_MODEL = "minimax/MiniMax-M3"
_RECALL_TIMEOUT = 20.0
_STORE_TIMEOUT = 30.0
_EXTRACT_TIMEOUT = 60.0
_PREFETCH_WAIT = 6.0

BRAIN_GRAPH_SCHEMA: Dict[str, Any] = {
    "name": "brain_graph",
    "description": (
        "Typed knowledge graph (subject —predicate→ object). Relevant facts are "
        "recalled automatically each turn; use this tool to TRAVERSE links or to "
        "store a relation explicitly.\n"
        "ACTIONS:\n"
        "• recall — graph search: matching facts + their 1-hop neighbours\n"
        "• neighbors — everything directly connected to an entity\n"
        "• path — how two entities are connected (chain of relations)\n"
        "• god_nodes — the most connected entities (memory hubs)\n"
        "• communities — thematic clusters of entities\n"
        "• entity — all facts about one entity\n"
        "• store — add a relation: subject/predicate/object are canonical English, "
        "fact/context free-form. If you can't split it, just pass the sentence in "
        "`fact` — it is captured immediately and linked up in the background.\n"
        "• search — plain semantic search over facts\n"
        "• delete — remove a WRONG fact. Pass `fact_id`, or `fact` with the EXACT "
        "stored sentence. A non-exact `fact` is refused and the tool lists candidate "
        "ids — retry with `fact_id`.\n"
        "• update — replace a fact's sentence: `fact_id` (or the EXACT `fact`) + "
        "`content`. A non-exact `fact` is refused with candidate ids — retry with "
        "`fact_id`.\n"
        "IDs: recall / search / neighbors print every fact as `[abc12345] …` — that "
        "bracketed prefix IS the `fact_id`. To fix a recalled fact, copy its id and "
        "call update/delete with `fact_id`. NEVER edit the database directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["recall", "neighbors", "path", "god_nodes",
                         "communities", "entity", "store", "search",
                         "delete", "update"],
            },
            "query": {"type": "string", "description": "Query for 'recall'/'search'."},
            "entity": {"type": "string", "description": "Entity name for 'neighbors'/'entity'."},
            "from": {"type": "string", "description": "Start entity for 'path'."},
            "to": {"type": "string", "description": "End entity for 'path'."},
            "subject": {"type": "string", "description": "Subject for 'store' (canonical English)."},
            "predicate": {"type": "string", "description": "Relation verb for 'store' (is/has/uses/runs)."},
            "object": {"type": "string", "description": "Object for 'store' (canonical English)."},
            "fact": {"type": "string", "description": "Full sentence for 'store'."},
            "context": {"type": "string", "description": "Why it matters, for 'store'."},
            "limit": {"type": "integer", "description": "Max results (god_nodes/recall)."},
            "fact_id": {"type": "string", "description": "Fact id (or its prefix) for 'delete'/'update'."},
            "content": {"type": "string", "description": "New sentence for 'update'."},
        },
        "required": ["action"],
    },
}

_EXTRACT_PROMPT = """You extract knowledge-graph triples from a conversation turn.

Return ONLY a JSON array (no prose, no code fence). Each item:
{"subject": "...", "predicate": "...", "object": "...", "fact": "...", "context": "..."}

WHAT TO EXTRACT — the guiding principle: anything the user would expect you to still
know tomorrow, or would be annoyed to explain twice. That explicitly includes:
- who/what things are and how they relate to each other
- events and transactions: purchases, payments, sums, prices, weights, dates
- quantities and states: what is owned, consumed, available, planned
- preferences, decisions, commitments
- setup and domain knowledge: servers, configs, accounts, rules of a craft
When unsure, extract it — a missed fact costs more than a redundant one.

WHAT TO SKIP — only content-free conversational filler: greetings, thanks,
acknowledgements ("ok", "\u0441\u043f\u0430\u0441\u0438\u0431\u043e", "\u0434\u043e\u0431\u0440\u0435"), and questions that assert nothing new.

WHOSE FACT IS IT - the reliability rule, and it overrides everything above:
The graph is read a year from now by someone who never saw this conversation. So a
triple must be a fact about the world or the user, never about this dialogue or
about you. If a statement stops making sense once the conversation is forgotten, it
is not a fact - drop it.
  DROP: what you noticed, missed, assumed, corrected or lack access to; the state of
        the conversation; that a message or file "had an error"; your own apologies.
        Never make yourself, the assistant, or the conversation a subject or object.

The user's words are evidence. Your own reading of an image, document or a guess is
NOT evidence - it is a hypothesis, and hypotheses must not enter long-term memory.
From the Assistant turn, extract only what restates or arithmetically follows from
what the user actually provided. Anything you inferred yourself - above all the
identity labels you read off a photo: store, brand, city, country, date, currency,
document number - must be left out unless the user wrote it in text or confirmed it.
  A wrong identity label is worse than a missing one: it silently reattributes real
  numbers to a place or time they never belonged to, and nothing later contradicts it.

FORMAT - these are hard rules, follow them exactly:
- subject/object MUST be English, even when the conversation is in another language.
  Translate names: "\u043a\u0443\u0440\u0447\u0430 \u0444\u0456\u043b\u0435" -> "chicken fillet",
  "\u0442\u0438\u043b\u0430\u043f\u0456\u044f" -> "tilapia", "\u0421\u0456\u043b\u044c\u043f\u043e" -> "silpo".
- A subject/object is a STABLE IDENTIFIER: never put quantities, prices, dates or
  units inside it. Those belong in separate triples.
    CORRECT:   {"subject":"chicken fillet","predicate":"weighs","object":"1.534 kg"}
               {"subject":"chicken fillet","predicate":"costs","object":"309 uah per kg"}
    WRONG:     {"subject":"chicken fillet 1.534 kg at 309 uah/kg", ...}
    WRONG:     {"subject":"800 g tilapia", ...}   (use "tilapia")
- A triple connects two DIFFERENT things: the same name must never appear on both
  sides. When the statement is about someone owning, needing or planning something,
  that person is the subject.
    CORRECT:   {"subject":"oleksandr","predicate":"needs to buy","object":"milk"}
    WRONG:     {"subject":"milk","predicate":"planned to buy","object":"milk"}
- lowercase, except proper names (Anyuta, ASUS, AdGuard Home)
- predicate: short English verb phrase (is, has, runs, uses, bought, paid, costs,
  weighs, lives in, prefers, plans)
- fact: one full sentence in the speaker's language
- context: short phrase - why it matters
- At most 8 triples. Only if the turn truly asserts nothing, return [].

CONVERSATION TURN:
User: {user}
Assistant: {assistant}
"""



_BREATH_TURNS = 10        # выдох по числу ходов
_BREATH_SECONDS = 7200    # ...или по времени (2 часа)
_PERIOD_CHARS = 20000     # верхний предел периода в символах (MiniMax-M3 держит)

_CONSOLIDATE_PROMPT = """You consolidate a PERIOD of conversation into long-term graph memory.

Return ONLY a JSON array. Each item:
{"subject": "...", "predicate": "...", "object": "...", "fact": "...", "context": "..."}

WHAT TO KEEP — the guiding principle: facts, relations and QUANTITATIVE DATA that
stay useful weeks from now. Especially preserve numbers the user reported —
measurements, weights, prices, sums, dates, composition: they are the basis for
future calculations (diet, budget, macros). Generalise many small events into a
fact, BUT never drop the numbers while generalising. Time that marks a routine
(dinner at 22:00, measured in the morning) is data — keep it. Do NOT claim a
recurring pattern from a single event.

WHAT TO SKIP — only the momentary WITHOUT data: notifications, dialogue state,
one-off statuses ("it is now 22:00", "drinking water now"). A line that carries a
number the user gave is data — keep it. Never make yourself or the conversation a
subject or object. Your own reading of an image is a hypothesis, not evidence:
identity labels off a photo (store, city, currency) only if the user stated them.

DO NOT invent numbers the user did not state.

FORMAT:
- subject/object MUST be English, stable identifiers (translate names). Never put
  quantities/prices/dates inside them — those go in separate triples (weighs /
  costs / measures) and in the `fact` sentence.
- lowercase except proper names; predicate = short English verb phrase.
- Build relations between STABLE entities so conclusions can be drawn later.

DECOMPOSE STRUCTURE — CRITICAL:
When the period contains a plan, schedule, list, table or multi-day layout (a weekly
menu, a shopping list, a set of steps), NEVER store it as one fact with everything
crammed into `fact`. Emit ONE triple per line / day / item, each with its own subject
(e.g. `meal-plan-2026-07-29`, `grocery-list-2026-08-01`) and a short object, so a later
question about a single day or item retrieves exactly that line — not the whole blob.
A monolithic multi-day fact is un-searchable by day and will be split downstream.

PERIOD:
{period}
"""

_DAY_MARK = re.compile(
    r"(?<![а-яіїєґ0-9])(пн|вт|ср|чт|пт|сб|нд|mon|tue|wed|thu|fri|sat|sun|"
    r"понеділ|вівтор|серед|четвер|п.ятниц|субот|неділ)[\s.,:)]",
    re.IGNORECASE)
_DAY_DATE = re.compile(
    r"(?=(?:пн|вт|ср|чт|пт|сб|нд)\s+\d{1,2}[.\-/]\d{1,2})", re.IGNORECASE)


def _slugify(text: str, n: int = 6) -> str:
    words = re.findall(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+", text.lower())[:n]
    return "-".join(words) or "item"


def _split_monolithic(fact_text: str) -> List[str]:
    """Многодневный/многопунктовый блоб -> список отдельных строк-фактов; иначе [].

    Ловит план/список, слитый моделью в один fact. Узкий триггер, чтобы
    не задеть обычные короткие однодневные факты."""
    raw = [ln.strip(" -\u2022*\t\u00b7\u2014") for ln in fact_text.splitlines()]
    lines = [ln for ln in raw
             if len(ln) > 8 and re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", ln)]
    # многострочный список/таблица (>=4 содержательных строк) — бьём всегда
    if len(lines) >= 4:
        return lines
    # однострочная простыня-план по дням — только если длинная
    if len(fact_text) > 250:
        day_hits = len({m.lower() for m in _DAY_MARK.findall(fact_text)})
        if day_hits >= 3:
            parts = [p.strip(" -\u2013\u2014.;") for p in _DAY_DATE.split(fact_text)]
            parts = [p for p in parts if len(p) > 8]
            if len(parts) >= 2:
                return parts
    return []


class BrainMemoryProvider(MemoryProvider):
    """Graph memory over the `brain` SPO store, wired into the turn loop."""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._session_id: str = ""
        self._cli_dir: str = _DEFAULT_CLI_DIR
        self._db: str = _DEFAULT_DB
        self._python: str = ""
        self._cli: str = ""
        self._llm_base_url: str = ""
        self._llm_api_key: str = ""
        self._extract_model: str = _DEFAULT_EXTRACT_MODEL
        self._auto_store: bool = True
        self._recall_limit: int = 8
        # prefetch cache (mirrors mem0: never block the turn on a slow backend)
        self._pf_lock = threading.Lock()
        self._pf_query: str = ""
        self._pf_result: str = ""
        self._pf_done: bool = False
        self._pf_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()
        self._breath_lock = threading.Lock()
        self._breath_path: str = ""

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "brain"

    def is_available(self) -> bool:
        self._resolve_paths()
        return bool(self._python and os.path.exists(self._python)
                    and self._cli and os.path.exists(self._cli))

    def _resolve_paths(self) -> None:
        self._cli_dir = str(self._config.get("cli_dir") or _DEFAULT_CLI_DIR)
        self._db = str(self._config.get("brain_db") or os.environ.get("BRAIN_DB") or _DEFAULT_DB)
        self._python = os.path.join(self._cli_dir, ".venv", "bin", "python")
        self._cli = os.path.join(self._cli_dir, "cli.py")

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._config = dict(self._load_plugin_config())
        self._resolve_paths()
        self._auto_store = bool(self._config.get("auto_store", True))
        self._extract_model = str(self._config.get("extract_model") or _DEFAULT_EXTRACT_MODEL)
        self._recall_limit = int(self._config.get("recall_limit", 8) or 8)
        self._llm_base_url, self._llm_api_key = self._resolve_llm()
        _home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        self._breath_path = os.path.join(_home, f"brain_breath_{self._db}.jsonl")
        logger.info(
            "brain graph memory: db=%s auto_store=%s extract_model=%s breath=%s",
            self._db, self._auto_store, self._extract_model, self._breath_path,
        )

    def _load_plugin_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            return (cfg.get("plugins") or {}).get("hermes-brain") or {}
        except Exception:
            return {}

    def _resolve_llm(self) -> tuple[str, str]:
        """Reuse the agent's own chat endpoint (OmniRoute for Paul) for extraction."""
        base = str(self._config.get("llm_base_url") or "")
        key = str(self._config.get("llm_api_key") or "")
        if base and key:
            return base, key
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            model = cfg.get("model") or {}
            return str(model.get("base_url") or ""), str(model.get("api_key") or "")
        except Exception:
            return "", ""

    def shutdown(self) -> None:
        for t in (self._pf_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)

    # -- subprocess helper ---------------------------------------------------

    def _run_cli(self, args: List[str], timeout: float) -> str:
        env = dict(os.environ)
        env["BRAIN_DB"] = self._db
        try:
            proc = subprocess.run(
                [self._python, self._cli, *args],
                capture_output=True, text=True, timeout=timeout,
                env=env, cwd=self._cli_dir,
            )
            return (proc.stdout or "").strip()
        except subprocess.TimeoutExpired:
            logger.warning("brain cli '%s' timed out after %ss", args[0] if args else "?", timeout)
            return ""
        except Exception as e:
            logger.warning("brain cli '%s' failed: %s", args[0] if args else "?", e)
            return ""

    # -- system prompt -------------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "# Brain — your single long-term memory (knowledge graph)\n"
            f"Active (graph: {self._db}). This is your ONLY long-term store — there is no "
            "mem0. Facts live as **typed relations** `subject —predicate→ object` plus a "
            "vector, so brain both recalls by meaning AND answers how things connect.\n"
            "How it works:\n"
            "- **Automatic recall:** before each turn, matching relations *and their "
            "neighbours* are injected under `## Brain Graph`. Treat that block as true "
            "recollection — never say you have no memory when it is present.\n"
            "- **Breathing (automatic storage):** you do NOT distil every turn yourself. "
            "Turns are buffered and consolidated periodically (every ~10 turns / 2 hours) "
            "into durable facts, data and relations — routine chatter is summarised, "
            "momentary noise dropped. So never think 'nothing was saved' after one turn; "
            "the exhale collects it. Do not hand-save ordinary conversation.\n"
            "- **Write precise data NOW, don't wait for the exhale:** anything you must "
            "get exactly right in the moment — measurements, prices, inventory counts, "
            "a running tally ('ate 2 eggs → 16 left') — write immediately with "
            "`brain_graph action=store` (or `update` to change a count), because the "
            "next turn may depend on the exact number before breathing runs. Keep such "
            "state as relations: `pantry —has→ eggs`, fact carries the count.\n"
            "- **Traversal:** call `brain_graph` when a question is about *connections* — "
            "`neighbors` (what touches X), `path` (how X relates to Y), `god_nodes`, "
            "`communities`, `entity` (all about X), `recall` (vector + graph).\n"
            "- **Corrections are mandatory, and easy:** the graph is not append-only. "
            "Every recalled fact is shown with its id in brackets — `[abc12345] subject "
            "→ predicate → object`. To fix one, take that id and call `action=update "
            "fact_id=abc12345 content=...` (or `action=delete fact_id=abc12345`). You do "
            "NOT need the exact stored sentence, and you must NEVER open the database "
            "directly — the id from recall is all you need. When the user corrects you, "
            "or a recalled fact is wrong, fix it BEFORE answering; apologising without "
            "fixing leaves the error to be recalled tomorrow.\n"
            "- If a `store` result mentions SIMILAR FACTS, check whether the older one is "
            "now wrong; if so, delete it instead of keeping both.\n"
            "You never need an external file or ad-hoc script for tracking — brain IS the "
            "database: store state as relations and update them."
        )

    # -- auto-recall ---------------------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._start_prefetch(query)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _start_prefetch(self, query: str) -> None:
        if not query or not query.strip():
            return
        with self._pf_lock:
            if self._pf_query == query and (self._pf_done or
                                            (self._pf_thread and self._pf_thread.is_alive())):
                return
            self._pf_query, self._pf_result, self._pf_done = query, "", False

        def _run() -> None:
            body = self._recall_block(query)
            with self._pf_lock:
                if self._pf_query == query:
                    self._pf_result, self._pf_done = body, True

        t = threading.Thread(target=_run, daemon=True, name="brain-prefetch")
        with self._pf_lock:
            self._pf_thread = t
        t.start()

    def _consume_prefetch(self, query: str) -> Optional[str]:
        with self._pf_lock:
            if self._pf_query != query or not self._pf_done:
                return None
            out, self._pf_result, self._pf_done = self._pf_result, "", False
            return out

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cached = self._consume_prefetch(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        with self._pf_lock:
            thread = self._pf_thread if self._pf_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT)
        cached = self._consume_prefetch(query)
        return cached if cached is not None else ""

    def _recall_block(self, query: str) -> str:
        raw = self._run_cli(["recall", query], _RECALL_TIMEOUT)
        if not raw or "No matching facts" in raw:
            return ""
        lines: List[str] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("🧠"):
                continue
            if s.startswith("--") and s.endswith("--"):
                lines.append(s.strip("- ").strip())
                continue
            if s.startswith("["):
                lines.append("- " + s)
            elif lines:
                lines.append("  " + s)
            if len(lines) >= self._recall_limit * 3:
                break
        return "## Brain Graph\n" + "\n".join(lines) if lines else ""

    # -- auto-store ----------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # BREATHING: no longer distil every turn. Buffer the turn cheaply; the
        # LLM consolidator runs periodically (breathe) over the whole period so
        # memory keeps facts/data/relations, not a transcript of every turn.
        if not self._auto_store or not (user_content or assistant_content):
            return
        try:
            with self._breath_lock:
                with open(self._breath_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(
                        {"u": user_content, "a": assistant_content, "ts": time.time()},
                        ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("brain breath buffer write failed: %s", e)
            return
        if self._breath_due():
            self._breathe_async()

    def _breath_due(self) -> bool:
        try:
            with open(self._breath_path, encoding="utf-8") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        except FileNotFoundError:
            return False
        if len(lines) >= _BREATH_TURNS:
            return True
        if lines:
            try:
                first_ts = json.loads(lines[0]).get("ts", 0)
            except Exception:
                return True
            if time.time() - first_ts >= _BREATH_SECONDS:
                return True
        return False

    def _breathe_async(self) -> None:
        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                return  # already breathing
            self._sync_thread = threading.Thread(
                target=self.breathe, daemon=True, name="brain-breathe")
            self._sync_thread.start()

    def breathe(self) -> None:
        """Consolidate the buffered period into long-term facts + relations."""
        if not self._llm_base_url or not self._llm_api_key:
            return
        with self._breath_lock:
            try:
                with open(self._breath_path, encoding="utf-8") as fh:
                    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            except FileNotFoundError:
                return
            if not lines:
                return
            turns = []
            for ln in lines:
                try:
                    turns.append(json.loads(ln))
                except Exception:
                    continue
        if not turns:
            return
        period = "\n".join(
            f"User: {t.get('u','')}\nAssistant: {t.get('a','')}" for t in turns)[:_PERIOD_CHARS]
        # источник истины при дыхании — весь период (реплики обеих сторон),
        # т.к. консолидатор обобщает текст диалога, а не читает фото заново.
        user_words = " ".join((t.get("u","")+" "+t.get("a","")) for t in turns)
        try:
            triples = self._consolidate(period)
        except Exception as e:
            logger.warning("brain breathe: consolidation failed, keeping buffer: %s", e)
            return  # leave buffer — retry on next trigger
        kept = []
        for t in triples:
            missing = unsupported_precise_values(t, user_words)
            if missing:
                logger.info("brain breathe: dropped %r — invented numbers %s",
                            t.get("subject"), missing)
                continue
            kept.append(t)
        for t in kept:
            self._store_triple(t)
        logger.info("brain breathe: %d turn(s) -> %d fact(s)", len(turns), len(kept))
        # success — clear only the turns we consumed (append-safe: rewrite remainder)
        with self._breath_lock:
            try:
                with open(self._breath_path, encoding="utf-8") as fh:
                    all_lines = [ln for ln in fh.read().splitlines() if ln.strip()]
                remainder = all_lines[len(lines):]
                with open(self._breath_path, "w", encoding="utf-8") as fh:
                    fh.write(("\n".join(remainder) + "\n") if remainder else "")
            except Exception as e:
                logger.warning("brain breathe: buffer clear failed: %s", e)

    def _consolidate(self, period: str) -> List[Dict[str, str]]:
        import httpx
        prompt = _CONSOLIDATE_PROMPT.replace("{period}", period)
        payload = {
            "model": self._extract_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }
        with httpx.Client(timeout=_EXTRACT_TIMEOUT) as client:
            r = client.post(
                self._llm_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self._llm_api_key}"},
                json=payload,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        return self._parse_triples(text)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Exhale whatever is buffered when the session closes.
        try:
            self.breathe()
        except Exception as e:
            logger.debug("brain on_session_end breathe skipped: %s", e)

    def _extract_triples(self, user: str, assistant: str) -> List[Dict[str, str]]:
        import httpx

        prompt = _EXTRACT_PROMPT.replace("{user}", user[:4000]).replace(
            "{assistant}", assistant[:4000])
        payload = {
            "model": self._extract_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }
        with httpx.Client(timeout=_EXTRACT_TIMEOUT) as client:
            r = client.post(
                self._llm_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self._llm_api_key}"},
                json=payload,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        return self._parse_triples(text)

    @staticmethod
    def _parse_triples(text: str) -> List[Dict[str, str]]:
        if not text:
            return []
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except Exception:
            return []
        out: List[Dict[str, str]] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            subj, pred, obj = (str(item.get(k, "")).strip()
                               for k in ("subject", "predicate", "object"))
            if not (subj and pred and obj):
                continue
            out.append({
                "subject": subj, "predicate": pred, "object": obj,
                "fact": str(item.get("fact", "")).strip() or f"{subj} {pred} {obj}",
                "context": str(item.get("context", "")).strip() or "auto-extracted from conversation",
            })
        return out[:5]

    def _store_triple(self, t: Dict[str, str], confidence: str = "extracted") -> str:
        # СТРАХОВКА: модель обязана раскладывать план/список на связи, но через раз
        # сливает всё в один многодневный fact. Ловим монолит и бьём построчно/по-дням,
        # чтобы recall по конкретному дню/пункту находил точную запись, а не блоб.
        parts = _split_monolithic(str(t.get("fact", "")))
        if parts:
            logger.info("brain: monolithic fact split into %d parts (subject=%r)",
                        len(parts), t.get("subject"))
            results = []
            for ln in parts:
                results.append(self._raw_store({
                    "subject": t.get("subject", ""),
                    "predicate": t.get("predicate", "") or "includes",
                    "object": _slugify(ln),
                    "fact": ln,
                    "context": t.get("context", ""),
                }, confidence))
            return " ; ".join(r for r in results if r)[:300]
        return self._raw_store(t, confidence)

    def _raw_store(self, t: Dict[str, str], confidence: str = "extracted") -> str:
        # страховка от самопетель: ребро "X -> X" знания не несёт
        if str(t.get("subject", "")).strip().lower() == str(t.get("object", "")).strip().lower():
            logger.debug("brain: skipped self-loop triple %r", t.get("subject"))
            return "skipped self-loop"
        out = self._run_cli(
            ["store", t["subject"], t["predicate"], t["object"],
             t["fact"], t["context"], "", confidence, "-y"],
            _STORE_TIMEOUT,
        )
        # brain prints the near-duplicates it found before force-storing. Surface them:
        # a "similar" fact with different numbers is usually an outdated statement the
        # agent should delete, so make it visible instead of silently piling both up.
        if "SIMILAR FACTS FOUND" in out:
            logger.info("brain: possible contradiction while storing %r -> %s",
                        t["fact"][:80],
                        " / ".join(ln.strip() for ln in out.splitlines()
                                   if ln.strip().startswith("["))[:300])
        return out

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [BRAIN_GRAPH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "brain_graph":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        action = str(args.get("action", "")).strip()

        def ok(out: str) -> str:
            return json.dumps({"result": out or "No results."}, ensure_ascii=False)

        try:
            if action == "recall":
                q = str(args.get("query", "")).strip()
                return ok(self._run_cli(["recall", q], _RECALL_TIMEOUT)) if q else \
                    json.dumps({"error": "Missing 'query'"})
            if action == "search":
                q = str(args.get("query", "")).strip()
                return ok(self._run_cli(["search", q], _RECALL_TIMEOUT)) if q else \
                    json.dumps({"error": "Missing 'query'"})
            if action in ("neighbors", "entity"):
                e = str(args.get("entity", "")).strip()
                return ok(self._run_cli([action, e], _RECALL_TIMEOUT)) if e else \
                    json.dumps({"error": "Missing 'entity'"})
            if action == "path":
                a, b = str(args.get("from", "")).strip(), str(args.get("to", "")).strip()
                return ok(self._run_cli(["path", a, b], _RECALL_TIMEOUT)) if a and b else \
                    json.dumps({"error": "Need 'from' and 'to'"})
            if action == "god_nodes":
                limit = str(int(args.get("limit", 10) or 10))
                return ok(self._run_cli(["god_nodes", limit], _RECALL_TIMEOUT))
            if action == "communities":
                return ok(self._run_cli(["communities"], _RECALL_TIMEOUT))
            if action == "store":
                subj, pred, obj = (str(args.get(k, "")).strip()
                                   for k in ("subject", "predicate", "object"))
                if subj and pred and obj:
                    triple = {
                        "subject": subj, "predicate": pred, "object": obj,
                        "fact": str(args.get("fact", "")).strip() or f"{subj} {pred} {obj}",
                        "context": str(args.get("context", "")).strip() or "stated by the user",
                    }
                    return ok(self._store_triple(triple, confidence="stated"))
                # СТРАХОВКА (guard-in-code): модель через раз зовёт store ПЛОСКО, как
                # файловый memory-tool — кладёт факт в `content`/`fact`, ключ в `fact_id`,
                # мета в `context`, а predicate/object не даёт. Раньше это отбивалось
                # "Need subject/predicate/object" и факт утекал в MEMORY.md.
                #
                # НЕ зовём LLM здесь: синхронный декомпозер конкурировал с основным ходом
                # на одном MiniMax и таймаутил 60с × ~10 store/ход. Пишем факт МГНОВЕННО и
                # детерминированно (запись памяти не должна зависеть от флаки-эндпоинта).
                # Осмысленные связи из того же разговора достраивает «дыхание» в фоне —
                # ему LLM доступен вне критического пути.
                #
                # ВАЖНО: эта модель кладёт сам ФАКТ в `context` (как файловый memory-tool
                # кладёт содержимое в `content`), а `fact_id` использует как ключ. Напр.
                # {"context":"31.07 сніданок: 250 г гречки...","fact_id":"meal-1-..."}.
                # Поэтому текст факта берём из ЛЮБОГО из fact/content/context (самый длинный),
                # иначе получались заглушки "subject records item" без содержания.
                free = max((str(args.get(k, "")).strip()
                            for k in ("fact", "content", "context")),
                           key=len, default="")
                fid = str(args.get("fact_id", "")).strip()
                if not (free or subj or fid):
                    return json.dumps({"error": "Need at least a 'subject'/'fact_id' "
                                       "or a 'fact'/'content'/'context' text"})
                # если текста нет вовсе — хоть де-слагнутый ключ, чтобы факт был находим
                if not free:
                    free = (fid or subj).replace("-", " ").replace("_", " ").strip()
                subj = subj or _slugify(fid) or _slugify(free)
                pred = pred or "records"
                # object обязан отличаться от subject (self-loop пропускается): слаг факта,
                # иначе слаг ключа.
                obj = obj or _slugify(free) or _slugify(fid) or "note"
                if obj == subj:
                    obj = _slugify(fid) if _slugify(fid) != subj else "note"
                triple = {
                    "subject": subj, "predicate": pred, "object": obj,
                    "fact": free,
                    # context факта = ключ модели (для трассировки), не дублируем текст
                    "context": fid or "stated by the user",
                }
                return ok(self._store_triple(triple, confidence="stated"))
            if action == "delete":
                fid = str(args.get("fact_id", "")).strip()
                text = str(args.get("fact", "")).strip()
                if fid:
                    return ok(self._run_cli(["delete", "--id", fid, "-y"], _STORE_TIMEOUT))
                if not text:
                    return json.dumps({"error": "Need 'fact_id' or exact 'fact' text"})
                return ok(self._run_cli(["delete", text, "-y"], _STORE_TIMEOUT))
            if action == "update":
                fid = str(args.get("fact_id", "")).strip()
                new = str(args.get("content", "")).strip()
                text = str(args.get("fact", "")).strip()
                if not new:
                    return json.dumps({"error": "Need 'content' (the corrected sentence)"})
                cmd = ["update"]
                if fid:
                    cmd += ["--id", fid]
                elif text:
                    cmd += [text]
                else:
                    return json.dumps({"error": "Need 'fact_id' or exact 'fact' text"})
                cmd += ["--content", new, "-y"]
                return ok(self._run_cli(cmd, _STORE_TIMEOUT))
            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            return json.dumps({"error": f"brain_graph failed: {e}"}, ensure_ascii=False)

    # -- config surface ------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "brain_db", "description": "Graph database name", "default": _DEFAULT_DB},
            {"key": "cli_dir", "description": "Directory with brain cli.py and .venv",
             "default": _DEFAULT_CLI_DIR},
            {"key": "auto_store", "description": "Distil every turn into triples",
             "default": "true", "choices": ["true", "false"]},
            {"key": "extract_model", "description": "Model used for triple extraction",
             "default": _DEFAULT_EXTRACT_MODEL},
            {"key": "recall_limit", "description": "Max facts injected per turn", "default": "8"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            from hermes_cli.config import load_config, save_config as _save
            cfg = load_config() or {}
            plugins = cfg.setdefault("plugins", {})
            section = plugins.setdefault("hermes-brain", {})
            section.update(values)
            _save(cfg)
        except Exception as e:
            logger.warning("brain save_config failed: %s", e)


def register(ctx) -> None:
    """Plugin entry point — hand the provider to hermes."""
    ctx.register_memory_provider(BrainMemoryProvider())
