"""latency_logger.py — a dedicated, self-contained timing/latency log for the
live PersonaPlex + IMTalker pipeline.

Why a separate file at all: `conversation_logger.py` answers "what happened and
why"; this answers "where did the seconds go". Mixing the two made per-stage
timings impossible to read -- the timing lines were scattered between
transcripts, search results and narrative paragraphs, and the only end-to-end
number available was `turn_done`, which stops at the moment the grounding text
was injected, i.e. BEFORE the model had said a single word.

Two files per server run, both append-only:

  1. `latency_<session>.log`  -- human-readable. A live line per stage as it
     completes, then one self-contained block per turn: a timeline of when each
     component finished (as an offset from the moment the user stopped
     speaking), the stage durations sorted slowest-first, the token counts
     (including how many tokens the compressor produced), and the headline
     question->first word / question->finished answer latencies.
  2. `latency_<session>.jsonl` -- one JSON object per finished turn, with every
     stage, mark, counter and headline metric as a flat field, for plotting or
     regression-checking a change against a previous run.

Everything here is best-effort: a logging failure must never break the live
pipeline, so every public method swallows its own exceptions. Thread-safety:
`start_turn`/`mark`/`stage`/`count` are called from both the GPU thread and the
background routing/search thread, so all state mutation and file I/O happens
under one lock.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

# Human-readable names for every stage/mark the pipeline reports. Anything not
# listed still logs fine -- it just shows up under its raw name.
STAGE_LABELS: dict[str, str] = {
    "stt_decode": "speech-to-text decode (audio -> transcript)",
    "transcript_check": "transcript sanity check (script/language)",
    "rule_route": "quick rule check: is a search needed?",
    "router_model": "router model decision: is a search needed?",
    "decision": "search/no-search decision (total)",
    "web_search": "online search (network round trip)",
    "search_filter": "score, filter and sort the search results",
    "compression": "compress results into one spoken sentence (legacy combined stage)",
    "compression_extractive": "extract best sentence from results (no LLM, ~0ms)",
    "compression_llm": "compress results into one spoken sentence (LLM)",
    "compression_fallback": "multi-sentence extractive fallback (LLM produced nothing usable)",
    # -- Avatar/video pipeline. logs_2 showed STT (0.4ms), routing (0.0ms) and
    # the model's first audible frame (+0.05s) were all fast while the answer
    # still felt seconds late, which puts the delay squarely in these stages.
    "avatar_fm": "avatar: flow-matching motion generation (GPU)",
    "avatar_helium": "avatar: helium feature extraction (GPU)",
    "avatar_render_jpeg": "avatar: render frames + JPEG encode (GPU)",
    "avatar_first_chunk_publish": "avatar: FIRST chunk generated -> queued for sending",
    "ref_encode": "tokenize + trim the grounding text",
    "lookup_inject": "inject the <lookup> 'please wait' note",
    "ref_inject": "inject the grounding <ref> into the live context",
    "thinking_sound": "thinking sound played to cover the wait",
    "gen_lm": "answer generation: Moshi LM forward passes (GPU)",
    "gen_audio_decode": "answer generation: Mimi audio decode (GPU)",
    "gen_audio_encode": "answer generation: Mimi input encode (GPU)",
    "avatar_motion": "avatar motion synthesis (Helium adapter + flow matching)",
    "avatar_render": "avatar render + JPEG encode",
}

MARK_LABELS: dict[str, str] = {
    "speech_end": "user stopped speaking (end of question)",
    "turn_start": "turn opened, routing begins",
    "decision_made": "search/no-search decision made",
    "search_done": "online search returned",
    "grounding_ready": "grounding sentence ready",
    "ref_injected": "grounding handed to the model",
    "lookup_injected": "'please wait' note handed to the model",
    "first_word": "assistant's FIRST spoken word (what the user feels)",
    "answer_complete": "assistant's LAST word of the answer",
    # -- Audio DELIVERY, kept strictly separate from text completion. --------
    # answer_complete only proves the model emitted its last TEXT token. These
    # three prove audio was generated, that it reached the websocket, and that
    # it finished. A turn with answer_complete but no audio_stream_started is
    # exactly the "text answer completes, but audio is never produced" failure.
    "first_audio_generated": "model produced its first non-silent audio frame",
    "audio_stream_started": "first audio packet actually written to the websocket",
    "audio_stream_completed": "last audio packet written to the websocket",
    "thinking_sound_started": "waiting/notification sound began",
    "thinking_sound_stopped": "waiting/notification sound ended",
}

# Headline metrics, in the order they are printed. (field, label)
HEADLINES: list[tuple[str, str]] = [
    ("question_to_decision_s", "question -> search decision"),
    ("question_to_search_done_s", "question -> search results in"),
    ("question_to_grounding_s", "question -> grounding sentence ready"),
    ("question_to_ref_injected_s", "question -> grounding given to model"),
    ("question_to_first_audio_s", "question -> first audio GENERATED"),
    ("question_to_audio_out_s", "question -> first audio SENT to browser"),
    ("generated_to_sent_s", "audio generated -> audio sent (pipeline delay)"),
    ("question_to_first_word_s", "question -> FIRST spoken word"),
    ("question_to_answer_complete_s", "question -> answer fully spoken"),
    ("answer_speaking_s", "duration of the spoken answer"),
]

_MAX_RETAINED_TURNS = 16


class TurnLatency:
    """Mutable timing record for one question -> answer cycle."""

    def __init__(self, turn_id: Any, t0: float, transcript: str = "") -> None:
        self.turn_id = turn_id
        self.transcript = transcript
        self.t0 = float(t0)
        self.wall_start = time.time()
        self.stages: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.counters: dict[str, Any] = {}
        self.notes: dict[str, str] = {}
        self.outcome = ""
        self.response = ""
        self.closed = False

    def offset(self, now: float | None = None) -> float:
        return max(0.0, (now if now is not None else time.perf_counter()) - self.t0)


class LatencyLogger:
    def __init__(self, log_dir: str = "", session_id: str = "", mirror_logger: Any = None) -> None:
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = str(log_dir or "")
        self.path: str | None = None
        self.jsonl_path: str | None = None
        # Optional stdlib `logging.Logger` (conversation_logger.py's own
        # logger) that every line written below is ALSO sent to, via
        # `logger.info(...)`. This is what makes latency/timing data show up
        # in the unified logs/conversation.log file -- see
        # runtime_logging.attach_conversation_file -- without this module
        # duplicating any of conversation_logger.py's file-handling logic.
        self._mirror = mirror_logger
        self._lock = threading.RLock()
        self._turns: dict[Any, TurnLatency] = {}
        self._order: list[Any] = []
        # Session-wide roll-up, printed with every turn block so a drift over a
        # long conversation is visible without post-processing the file.
        self._n_turns = 0
        self._sum_first_word = 0.0
        self._n_first_word = 0
        self._sum_total = 0.0
        self._n_total = 0
        self._n_searched = 0

        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.path = os.path.join(self.log_dir, f"latency_{self.session_id}.log")
            self.jsonl_path = os.path.join(self.log_dir, f"latency_{self.session_id}.jsonl")
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    "=" * 88 + "\n"
                    f"Latency / timing log -- session {self.session_id}\n"
                    f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    "Every timing in this file is measured from the moment the user STOPPED\n"
                    "speaking (t=0, when the question is complete) to the moment the assistant\n"
                    "finished speaking its answer. Lines marked 'stage' are measured durations of\n"
                    "one component; lines marked 'mark' are the instants a component finished.\n"
                    "Nothing here is ever overwritten -- each turn is appended below the last.\n"
                    + "=" * 88 + "\n\n"
                )
            print(
                f"[latency_logger] logging per-stage timings to {self.path} and {self.jsonl_path}",
                flush=True,
            )
        except Exception as e:  # pragma: no cover - logging must never break the pipeline
            print(f"[latency_logger] disabled, could not open log files: {e!r}", flush=True)
            self.path = None
            self.jsonl_path = None

    # -- low-level ---------------------------------------------------------

    @staticmethod
    def _now_ts() -> str:
        now = time.time()
        return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"

    def _write(self, text: str) -> None:
        if self._mirror is not None:
            try:
                stripped = text.strip("\n")
                if stripped:
                    self._mirror.info(stripped, extra={"conversation_id": self.session_id})
            except Exception:
                pass  # mirroring must never break the pipeline or the primary log below
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[latency_logger] failed to write log line: {e!r}", flush=True)

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self.jsonl_path:
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[latency_logger] failed to write jsonl record: {e!r}", flush=True)

    # -- turn lifecycle ----------------------------------------------------

    def start_turn(self, turn_id: Any, t0: float | None = None, transcript: str = "") -> None:
        """Open a record for `turn_id`. `t0` should be the perf_counter reading
        taken when the user's utterance ENDED (VAD fired) -- not when this call
        happens -- so that every offset below is measured from the instant the
        question was complete, which is what the user experiences as "I asked,
        then I waited"."""
        try:
            with self._lock:
                rec = TurnLatency(turn_id, t0 if t0 is not None else time.perf_counter(), transcript)
                self._turns[turn_id] = rec
                self._order.append(turn_id)
                # Keep only the most recent handful: a turn is finalized when
                # the NEXT one starts, so at most two are ever live, but a turn
                # that never produced a response would otherwise leak.
                while len(self._order) > _MAX_RETAINED_TURNS:
                    self._turns.pop(self._order.pop(0), None)
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{0.0:6.3f}s | START    | "
                f'question: "{transcript}"\n'
            )
        except Exception:
            pass

    def get(self, turn_id: Any) -> TurnLatency | None:
        with self._lock:
            return self._turns.get(turn_id)

    def stage(self, turn_id: Any, name: str, seconds: float, note: str = "") -> None:
        """Record a measured duration for one component and echo it live."""
        try:
            secs = float(seconds)
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                rec.stages[name] = rec.stages.get(name, 0.0) + secs
                if note:
                    rec.notes[name] = note
                offset = rec.offset()
            suffix = f" | {note}" if note else ""
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{offset:6.3f}s | stage    | "
                f"{name:<20} {secs * 1000.0:8.1f} ms{suffix}\n"
            )
        except Exception:
            pass

    def accumulate(self, turn_id: Any, name: str, seconds: float) -> None:
        """Add to a stage total WITHOUT writing a line. For per-80ms-chunk costs
        (LM forward, audio decode, avatar render) that would otherwise emit
        thousands of lines per turn; they are reported once in the turn block."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                rec.stages[name] = rec.stages.get(name, 0.0) + float(seconds)
        except Exception:
            pass

    def mark(self, turn_id: Any, name: str, note: str = "", at: float | None = None) -> None:
        """Record the instant a component finished, as an offset from t=0."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                offset = rec.offset(at)
                rec.marks[name] = offset
                if note:
                    rec.notes[name] = note
            suffix = f" | {note}" if note else ""
            self._write(
                f"[{self._now_ts()}] TURN {turn_id} | +{offset:6.3f}s | mark     | {name}{suffix}\n"
            )
        except Exception:
            pass

    def count(self, turn_id: Any, **counters: Any) -> None:
        """Record counts (tokens, results, characters) for this turn."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None:
                    return
                clean = {k: v for k, v in counters.items() if v is not None}
                rec.counters.update(clean)
            if clean:
                body = " ".join(f"{k}={v}" for k, v in clean.items())
                self._write(
                    f"[{self._now_ts()}] TURN {turn_id} | {'':>7} | count    | {body}\n"
                )
        except Exception:
            pass

    def note_transcript(self, turn_id: Any, transcript: str) -> None:
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is not None and transcript:
                    rec.transcript = transcript
        except Exception:
            pass

    # -- finalization ------------------------------------------------------

    def finish_turn(self, turn_id: Any, response: str = "", outcome: str = "") -> None:
        """Close the turn and write its self-contained block + JSONL record.
        Safe to call twice (the second call is a no-op) and safe to call for a
        turn that was never opened."""
        try:
            with self._lock:
                rec = self._turns.get(turn_id)
                if rec is None or rec.closed:
                    return
                rec.closed = True
                rec.response = response or ""
                rec.outcome = outcome or rec.outcome
                metrics = self._headline_metrics(rec)
                self._n_turns += 1
                if "question_to_first_word_s" in metrics:
                    self._sum_first_word += metrics["question_to_first_word_s"]
                    self._n_first_word += 1
                if "question_to_answer_complete_s" in metrics:
                    self._sum_total += metrics["question_to_answer_complete_s"]
                    self._n_total += 1
                if rec.stages.get("web_search"):
                    self._n_searched += 1
                block = self._render_block(rec, metrics)
                record = self._render_record(rec, metrics)
                headline = self._render_console(rec, metrics)
            self._write(block)
            self._write_jsonl(record)
            print(headline, flush=True)
        except Exception as e:
            print(f"[latency_logger] failed to finalize turn {turn_id}: {e!r}", flush=True)

    @staticmethod
    def _headline_metrics(rec: TurnLatency) -> dict[str, float]:
        m: dict[str, float] = {}
        pairs = [
            ("question_to_decision_s", "decision_made"),
            ("question_to_search_done_s", "search_done"),
            ("question_to_grounding_s", "grounding_ready"),
            ("question_to_ref_injected_s", "ref_injected"),
            ("question_to_first_audio_s", "first_audio_generated"),
            ("question_to_audio_out_s", "audio_stream_started"),
            ("question_to_first_word_s", "first_word"),
            ("question_to_answer_complete_s", "answer_complete"),
        ]
        for field, mark in pairs:
            if mark in rec.marks:
                m[field] = round(rec.marks[mark], 3)
        # The gap nothing else measures: how long generated audio sat in the
        # avatar/chunking/prebuffer path before it was actually sent out.
        if "first_audio_generated" in rec.marks and "audio_stream_started" in rec.marks:
            m["generated_to_sent_s"] = round(
                max(0.0, rec.marks["audio_stream_started"] - rec.marks["first_audio_generated"]), 3
            )
        if "first_word" in rec.marks and "answer_complete" in rec.marks:
            m["answer_speaking_s"] = round(
                max(0.0, rec.marks["answer_complete"] - rec.marks["first_word"]), 3
            )
        return m

    def _render_block(self, rec: TurnLatency, metrics: dict[str, float]) -> str:
        w = 88
        lines = ["", "=" * w]
        # ASCII only in this block: it is read with `tail`/`cat` on hosts whose
        # console is not guaranteed to be UTF-8, and a mojibake dash in the
        # header is the first thing a reader sees.
        lines.append(f"TURN {rec.turn_id} -- where the time went")
        lines.append(f'Question : "{rec.transcript}"')
        if rec.response:
            lines.append(f'Answer   : "{rec.response}"')
        if rec.outcome:
            lines.append(f"Outcome  : {rec.outcome}")
        lines.append(
            f"Clock    : question ended at "
            f"{time.strftime('%H:%M:%S', time.localtime(rec.wall_start))}"
            f".{int(rec.wall_start % 1 * 1000):03d}"
        )
        lines.append("-" * w)

        lines.append("Timeline (seconds after the user stopped speaking):")
        if rec.marks:
            for name, offset in sorted(rec.marks.items(), key=lambda kv: kv[1]):
                label = MARK_LABELS.get(name, name)
                lines.append(f"  +{offset:7.3f}s  {label}")
        else:
            lines.append("  (no timeline marks were recorded for this turn)")

        lines.append("")
        lines.append("Component times (slowest first):")
        total_measured = sum(rec.stages.values())
        span = metrics.get(
            "question_to_answer_complete_s", metrics.get("question_to_first_word_s", 0.0)
        )
        if rec.stages:
            for name, secs in sorted(rec.stages.items(), key=lambda kv: kv[1], reverse=True):
                label = STAGE_LABELS.get(name, name)
                pct = f"{secs / span * 100.0:5.1f}%" if span > 0 else "    -"
                note = rec.notes.get(name, "")
                note_s = f"   [{note}]" if note else ""
                lines.append(f"  {secs:7.3f}s  {pct} of total   {label}{note_s}")
            lines.append(f"  {total_measured:7.3f}s          all measured components added up")
        else:
            lines.append("  (no component timings were recorded for this turn)")

        if rec.counters:
            lines.append("")
            lines.append("Sizes / token counts:")
            for name, value in rec.counters.items():
                lines.append(f"  {name:<32} {value}")

        lines.append("")
        lines.append("Headline latencies:")
        for field, label in HEADLINES:
            if field in metrics:
                lines.append(f"  {label:<40} {metrics[field]:7.3f}s")
        if "question_to_first_word_s" not in metrics:
            lines.append(
                "  (the assistant never produced audible speech for this turn -- "
                "no first-word latency exists)"
            )

        avg_first = self._sum_first_word / self._n_first_word if self._n_first_word else 0.0
        avg_total = self._sum_total / self._n_total if self._n_total else 0.0
        lines.append("")
        lines.append(
            f"Session so far: {self._n_turns} turn(s), {self._n_searched} with an online search, "
            f"average question->first word {avg_first:.2f}s, "
            f"average question->finished answer {avg_total:.2f}s"
        )
        lines.append("=" * w)
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_record(rec: TurnLatency, metrics: dict[str, float]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "turn_latency",
            "ts": time.time(),
            "turn": rec.turn_id,
            "wall_start": rec.wall_start,
            "transcript": rec.transcript,
            "response": rec.response,
            "outcome": rec.outcome,
        }
        record.update(metrics)
        for name, secs in rec.stages.items():
            record[f"stage_{name}_s"] = round(float(secs), 4)
        for name, offset in rec.marks.items():
            record[f"at_{name}_s"] = round(float(offset), 4)
        record.update(rec.counters)
        record["stages_total_s"] = round(sum(rec.stages.values()), 4)
        return record

    @staticmethod
    def _render_console(rec: TurnLatency, metrics: dict[str, float]) -> str:
        first = metrics.get("question_to_first_word_s")
        total = metrics.get("question_to_answer_complete_s")
        parts = [
            f"first_word={first:.2f}s" if first is not None else "first_word=n/a",
            f"answer_done={total:.2f}s" if total is not None else "answer_done=n/a",
        ]
        for name in ("router_model", "web_search", "compression", "compression_fallback"):
            if rec.stages.get(name):
                parts.append(f"{name}={rec.stages[name]:.2f}s")
        if "compressor_output_tokens" in rec.counters:
            parts.append(f"compressor_out_tok={rec.counters['compressor_output_tokens']}")
        return f"[LATENCY] turn {rec.turn_id}: " + " ".join(parts)

    # -- session close -----------------------------------------------------

    def session_summary(self) -> None:
        """Optional end-of-session roll-up. Never called automatically -- the
        server is normally killed rather than shut down cleanly -- but useful
        from a notebook or a signal handler."""
        try:
            avg_first = self._sum_first_word / self._n_first_word if self._n_first_word else 0.0
            avg_total = self._sum_total / self._n_total if self._n_total else 0.0
            self._write(
                "\n" + "=" * 88 + "\n"
                f"SESSION SUMMARY {self.session_id}\n"
                f"  turns logged                     {self._n_turns}\n"
                f"  turns with an online search      {self._n_searched}\n"
                f"  average question->first word     {avg_first:.3f}s\n"
                f"  average question->finished answer{avg_total:.3f}s\n"
                + "=" * 88 + "\n\n"
            )
        except Exception:
            pass
