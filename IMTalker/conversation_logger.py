"""conversation_logger.py — structured conversation logging for the live
PersonaPlex + IMTalker pipeline.

Three outputs, all append-only (never truncated/overwritten while the
process is alive; each server run gets its own timestamped session_id, so
older sessions' files are never touched either):

  1. Console (stdout, already captured into live_server.log by the launch
     notebook) -- short one-line-per-event summaries.
  2. `conversation_<session>.log` / `.jsonl` -- the same short events, as
     text and as machine-parseable JSON lines, for scripts/dashboards.
  3. `latency_<session>.log` / `.jsonl` -- the dedicated timing/latency file
     (see latency_logger.py, exposed here as `self.latency`): where every
     second between "the user stopped speaking" and "the assistant finished
     answering" was spent, per component, plus the token counts of what the
     compressor produced. Kept separate because interleaving per-stage timings
     with transcripts and narrative made both unreadable.
  4. `detailed_<session>.log` -- a plain-English, step-by-step narrative of
     every request: what the user said, what the assistant decided to do,
     whether it decided a web search was needed and why, every search
     attempted (with scores and running counts), why and how a summary was
     created, exactly what was fed to the model and how, what it answered,
     and the "thinking sound" start/stop/loop detail. Written so a
     non-technical reader can follow the whole story of a single request top
     to bottom without needing to know what a router, an embedding, or a
     token is.

Thread-safety: called from both the GPU thread and the background
routing/search thread (see MoshiOnlyEngineWithHidden._route_and_search in
the AHAudioPace script). `logging.Logger`/`Handler` are internally thread-safe; the JSONL
and detailed-narrative write paths take their own lock. The turn-machine
these are called from only ever has one request in flight at a time, so the
detailed log's events arrive in true chronological order -- no buffering or
reordering needed to keep the narrative readable.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import runtime_logging  # IMTalker/ is on sys.path by the time this runs


class _NullLatency:
    """Stand-in used when latency_logger.py cannot be imported. Accepts and
    ignores every method the pipeline calls, so no call site needs a guard."""

    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop

    def get(self, _turn_id: Any) -> None:
        return None


class ConversationLogger:
    def __init__(self, log_dir: str = "", session_id: str = ""):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = str(log_dir or "")
        self._jsonl_path: str | None = None
        self._detail_path: str | None = None
        self._lock = threading.Lock()

        # Running total for this server process (i.e. across the whole
        # session, not per-request) -- "how many times online search is
        # performed".
        self.web_search_count = 0

        self.logger = logging.getLogger(f"conversation.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            # Millisecond-precision, time-only timestamps (HH:mm:ss.SSS) --
            # requested explicitly so per-stage timing events (retrieval, web
            # search, compression, ...) can be correlated to the millisecond;
            # second-precision made it impossible to tell how much of a
            # multi-event burst each individual stage actually took.
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(
                logging.Formatter("[%(asctime)s.%(msecs)03d] [CONV] %(message)s", datefmt="%H:%M:%S")
            )
            self.logger.addHandler(console)

            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)
                text_path = os.path.join(self.log_dir, f"conversation_{self.session_id}.log")
                file_handler = logging.FileHandler(text_path, encoding="utf-8")
                file_handler.setFormatter(
                    logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
                )
                self.logger.addHandler(file_handler)
                self._jsonl_path = os.path.join(self.log_dir, f"conversation_{self.session_id}.jsonl")

                self._detail_path = os.path.join(self.log_dir, f"detailed_{self.session_id}.log")
                with open(self._detail_path, "a", encoding="utf-8") as f:
                    f.write(
                        "=" * 80 + "\n"
                        f"Detailed conversation log -- session {self.session_id}\n"
                        f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        "This file records the full story of every request in plain language:\n"
                        "what was said, what was searched, what was found, what was given to\n"
                        "the assistant, and what it answered. Nothing here is ever overwritten --\n"
                        "each new request is appended below the last.\n"
                        + "=" * 80 + "\n\n"
                    )

                print(
                    f"[conversation_logger] logging conversation events to {text_path}, "
                    f"{self._jsonl_path}, and {self._detail_path}",
                    flush=True,
                )

            # Fixed-path logs/conversation.log: unconditional (unlike the
            # per-session files above, which need log_dir set), rotating, and
            # unified across every server run -- this is what shows the full
            # USER -> SEARCH DECISION -> ... -> PERSONAPLEX RESPONSE flow with
            # a conversation_id on every line, independent of session naming.
            runtime_logging.attach_conversation_file(self.logger)

        # Dedicated latency/timing log, sharing this session's id and directory
        # so `conversation_<s>.log`, `detailed_<s>.log` and `latency_<s>.log`
        # always describe the same run and can be read side by side. Created
        # unconditionally: with no log_dir it degrades to a no-op object, so
        # call sites never need to check whether it exists. `mirror_logger`
        # makes every per-stage timing line ALSO land in the fixed
        # logs/conversation.log above, with no duplicated computation.
        # The import is guarded rather than assumed: this module is imported
        # from inside the live engine's constructor, so an older checkout
        # without latency_logger.py must lose the timing log, not the whole
        # server. _NullLatency swallows every call site below.
        try:
            from latency_logger import LatencyLogger  # IMTalker/ is on sys.path by the time this runs

            self.latency: Any = LatencyLogger(
                log_dir=self.log_dir, session_id=self.session_id, mirror_logger=self.logger,
            )
        except Exception as e:
            print(
                f"[conversation_logger] latency logging disabled ({e!r}) -- "
                f"conversation logging is unaffected",
                flush=True,
            )
            self.latency = _NullLatency()

    # -- low-level ---------------------------------------------------------

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self._jsonl_path:
            return
        record = dict(record)
        record.setdefault("ts", time.time())
        record.setdefault("session_id", self.session_id)
        line = json.dumps(record, default=str, ensure_ascii=False)
        try:
            with self._lock, open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            # Logging must never crash the live pipeline.
            print(f"[conversation_logger] failed to write jsonl event: {e!r}", flush=True)

    def event(self, kind: str, summary: str = "", **fields: Any) -> None:
        line = f"[{kind}] {summary}" if summary else f"[{kind}]"
        extra = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
        if extra:
            line = f"{line} {extra}"
        self.logger.info(line, extra={"conversation_id": self.session_id})
        self._write_jsonl({"kind": kind, "summary": summary, **fields})

    @staticmethod
    def _now_ts() -> str:
        """HH:mm:ss.SSS, millisecond precision -- needed to tell how long each
        individual pipeline stage actually took when several events land
        within the same second."""
        now = time.time()
        return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"

    def _write_detail(self, turn_id: Any, heading: str, body: list[str]) -> None:
        """Append one readable, timestamped section to the detailed
        narrative log. Never raises -- logging must never break the pipeline."""
        if not self._detail_path:
            return
        ts = self._now_ts()
        turn_label = f"Turn #{turn_id}" if turn_id is not None else "—"
        lines = [f"[{ts}] {turn_label} — {heading}"]
        for b in body:
            for sub in str(b).splitlines() or [""]:
                lines.append(f"    {sub}")
        lines.append("")
        text = "\n".join(lines) + "\n"
        try:
            with self._lock, open(self._detail_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[conversation_logger] failed to write detailed log: {e!r}", flush=True)

    # -- convenience wrappers matching the actual pipeline events (short,
    # machine-friendly console/JSONL events) --------------------------------

    # -- Aligned per-turn console/log trace ---------------------------------
    #
    # One line per pipeline stage, all sharing a "TURN <n> | STAGE | ..."
    # prefix so a whole turn can be read top-to-bottom, or grepped out of a
    # busy log with `grep 'TURN 3 |'`. Every line still carries its own
    # millisecond timestamp, which is what makes the per-stage cost readable
    # without adding up separate timing events.
    #
    # ASCII '|' rather than a box-drawing character on purpose: this goes to
    # stdout as well as to a UTF-8 file, and stdout is not guaranteed to be
    # UTF-8 on every host that might tail these logs.

    _STAGE_W = 8

    def _turn(self, turn_id: Any, stage: str, text: str, kind: str, **fields: Any) -> None:
        self.logger.info(
            f"TURN {turn_id} | {stage:<{self._STAGE_W}} | {text}",
            extra={"conversation_id": self.session_id},
        )
        self._write_jsonl({"kind": kind, "turn": turn_id, **fields})

    def turn_heard(
        self, turn_id: Any, transcript: str, token_ids: list | None = None,
        script_stats: dict | None = None,
    ) -> None:
        """What the STT model produced for this turn. Token ids are included in
        the JSONL (not the console line) so a suspicious transcript can be
        traced back to the ids that produced it."""
        self._turn(
            turn_id, "HEARD", f'"{transcript}"', "user_transcript",
            transcript=transcript, turn_epoch=turn_id,
            token_ids=(token_ids[:64] if token_ids else None),
            n_tokens=(len(token_ids) if token_ids else 0),
            script_stats=script_stats,
        )

    def turn_transcript_rejected(
        self, turn_id: Any, transcript: str, stats: dict, token_ids: list | None = None,
    ) -> None:
        """The transcript was discarded before routing because it is not in a
        script this STT model can produce. Logged loudly and with the raw ids:
        this is a real defect somewhere upstream, not a normal outcome, and it
        must not be silently swallowed."""
        reason = stats.get("reason") or f'{stats.get("non_latin_ratio", 0):.0%} non-Latin'
        self._turn(
            turn_id, "REJECT",
            f'unusable transcript ({reason}) -- discarded, no search, no injection: '
            f'"{transcript[:120]}"',
            "transcript_rejected",
            transcript=transcript, script_stats=stats, reject_reason=reason,
            token_ids=(token_ids[:64] if token_ids else None),
            n_tokens=(len(token_ids) if token_ids else 0),
        )

    def turn_decision(
        self, turn_id: Any, transcript: str, needs_search: bool, source: str,
        score: float, elapsed_s: float, reason: str = "",
    ) -> None:
        """The search/no-search verdict for one turn. `source` is 'rules' when
        the instant regex pre-pass resolved it, 'model' when the Qwen router
        did, and 'error' when the router failed and the turn defaulted to
        searching."""
        verdict = "SEARCH ONLINE" if needs_search else "ANSWER DIRECTLY"
        self._turn(
            turn_id, "DECIDE",
            f"{verdict:<15} via={source:<6} score={score:.3f} ({1000.0 * elapsed_s:.1f}ms)",
            "router_decision",
            transcript=transcript, needs_search=bool(needs_search), source=source,
            score=round(float(score), 4), elapsed_s=round(float(elapsed_s), 4), reason=reason,
        )
        if reason:
            self.logger.info(
                f"TURN {turn_id} | {'':<{self._STAGE_W}} | why: {reason}",
                extra={"conversation_id": self.session_id},
            )

    def turn_search(
        self, turn_id: Any, provider: str, n_found: int, n_kept: int, elapsed_s: float,
        status: str = "success",
    ) -> None:
        """`status` is one of 'success', 'empty_results', or 'timeout' -- see
        the heuristic in _route_and_search (web_search_query/_sync always
        return [] on failure, so this is inferred from the result count and
        elapsed time against the configured timeout, not a distinct error
        channel from search_helpers.py)."""
        self._turn(
            turn_id, "SEARCH",
            f"{provider}: {n_found} found, {n_kept} kept, status={status} ({elapsed_s:.2f}s)",
            "turn_search",
            provider=provider, n_found=n_found, n_kept=n_kept, status=status,
            elapsed_s=round(float(elapsed_s), 3),
        )

    def turn_ground(self, turn_id: Any, text: str, used_fallback: bool, source: str = "") -> None:
        how = source or ("extractive" if used_fallback else "compressed")
        body = f'({how}) "{text}"' if text else f"({how}) nothing usable produced"
        self._turn(turn_id, "GROUND", body, "turn_ground",
                   grounding=text, used_fallback=bool(used_fallback), source=source)

    def turn_done(self, turn_id: Any, outcome: str, total_s: float) -> None:
        """Closes the turn's decision phase. `outcome` is a short phrase such
        as 'grounded from web search' or 'answered from own knowledge'."""
        self._turn(turn_id, "DONE", f"{outcome} ({total_s:.2f}s from end of speech)",
                   "turn_done", outcome=outcome, total_s=round(float(total_s), 3))

    def turn_spoke(self, turn_id: Any, seconds_to_first_word: float, backlog_s: float = 0.0) -> None:
        """The assistant's first audible audio for this turn.

        THIS is the end-to-end latency the user actually experiences; do not
        read latency off the TRANSCRIPT line below, which is emitted much later
        (see turn_replied). `backlog_s` is the microphone queue depth at that
        moment: the producer runs at real time, so a backlog is a delay that
        will not shrink on its own, and it is by far the most common reason
        this number is large."""
        extra = f"  [mic backlog {backlog_s:.1f}s]" if backlog_s >= 0.5 else ""
        self._turn(
            turn_id, "SPOKE", f"first word {seconds_to_first_word:.2f}s after the question{extra}",
            "turn_spoke",
            seconds_to_first_word=round(float(seconds_to_first_word), 3),
            input_backlog_s=round(float(backlog_s), 3),
        )

    def turn_replied(self, turn_id: Any, response: str) -> None:
        """The text this turn ended up speaking.

        Deliberately NOT called "REPLIED" in the log any more. It is emitted
        when the NEXT utterance ends -- that is the first moment the turn's
        text is known to be complete -- so its timestamp reflects when the user
        next stopped talking, not when the assistant answered. Reading latency
        from it produced the phantom 19s/39s figures in conversation_log_1."""
        self._turn(
            turn_id, "SAID", f'"{response}"    (transcript closed once the user spoke again)',
            "assistant_reply", response=response,
        )

    def turn_flags(
        self, turn_id: Any, ref_lora_active: bool, avatar_streaming_active: bool,
        thinking_sound_played: bool = False,
    ) -> None:
        """One consolidated per-turn line for the flags that don't change
        mid-turn: whether the <lookup>/<ref> LoRA is loaded (it's a
        startup-time, all-turns-or-none setting, logged here per turn anyway
        so a single turn's log block is self-contained), and whether the
        avatar/video streaming pipeline was live for this turn."""
        self._turn(
            turn_id, "FLAGS",
            f"ref_lora_active={ref_lora_active} avatar_streaming_active={avatar_streaming_active} "
            f"thinking_sound_played={thinking_sound_played}",
            "turn_flags",
            ref_lora_active=bool(ref_lora_active),
            avatar_streaming_active=bool(avatar_streaming_active),
            thinking_sound_played=bool(thinking_sound_played),
        )

    def quick_gate_timing(self, elapsed_s: float) -> None:
        """The instant regex pre-pass done synchronously on the GPU thread to
        decide whether a turn even needs the model router -- timed separately
        from the (much larger) router/search/compression stages below it, so a
        slow regex pass can be told apart from a slow model call."""
        self.event("quick_gate", f"elapsed={elapsed_s:.3f}s")

    def turn_timing_summary(
        self, turn_id: Any, total_s: float, stages: dict[str, float]
    ) -> None:
        """One consolidated, big-to-small breakdown of every timed stage in
        this turn's pipeline, so the single slowest stage is visible without
        having to manually sum up the individual per-stage lines above it."""
        ordered = sorted(stages.items(), key=lambda kv: kv[1], reverse=True)
        breakdown = " ".join(f"{name}={secs:.3f}s" for name, secs in ordered)
        self.event(
            "turn_timing_summary", f"turn={turn_id} total={total_s:.3f}s {breakdown}",
            turn=turn_id, total_s=round(total_s, 3),
            **{f"{name}_s": round(secs, 3) for name, secs in stages.items()},
        )

    def no_response_warning(self, turn_id: Any, transcript: str, elapsed_s: float) -> None:
        """The pipeline reached the next user utterance without the model
        having produced ANY spoken text for the previous turn -- previously
        this just left a silent gap in the log with no trace at all."""
        self.event(
            "no_response_warning",
            f"turn={turn_id} produced no spoken response after {elapsed_s:.1f}s",
            turn=turn_id, transcript=transcript, elapsed_s=round(elapsed_s, 1),
        )

    def retrieval(self, transcript: str, source: str, hits: list[dict], elapsed_s: float) -> None:
        preview = [
            {"title": h.get("title", ""), "source": h.get("source"),
             "score": round(float(h.get("similarity_score", 0.0)), 4),
             "text": str(h.get("text", ""))[:200]}
            for h in hits
        ]
        self.event(
            "retrieval", f"source={source} n_hits={len(hits)} elapsed={elapsed_s:.3f}s",
            transcript=transcript, hits=preview,
        )

    def web_search(self, query: str, provider: str, n_results: int, elapsed_s: float, triggered_reason: str = "") -> None:
        self.event(
            "web_search", f"provider={provider} n_results={n_results} elapsed={elapsed_s:.3f}s",
            query=query, triggered_reason=triggered_reason,
        )

    def compressor_call(self, question: str, passages: list[str], result: str, elapsed_s: float, used_fallback: bool) -> None:
        self.event(
            "compressor",
            f"elapsed={elapsed_s:.3f}s fallback={used_fallback}",
            question=question, passages=[p[:200] for p in passages], result=result,
        )

    def ref_injected(self, ref_text: str, n_tokens: int, elapsed_s: float, kind: str = "ref") -> None:
        self.event(
            f"{kind}_injected", f"n_tokens={n_tokens} elapsed={elapsed_s:.3f}s",
            text=ref_text,
        )

    def assistant_response(self, transcript: str, response_text: str) -> None:
        self.event("assistant_response", response_text, user_transcript=transcript)

    def component_status(self, **fields: Any) -> None:
        self.event("component_status", **fields)

    def error(self, where: str, exc: BaseException, traceback_text: str = "") -> None:
        self.event("error", f"{where}: {exc!r}", where=where, traceback=traceback_text[-4000:])

    # -- detailed, plain-English narrative (the new "any user can read it"
    # log) ------------------------------------------------------------------

    def narrate_user_message(self, turn_id: Any, transcript: str) -> None:
        self._write_detail(turn_id, "User spoke", [f'The user said: "{transcript}"'])

    def narrate_router_decision(
        self, turn_id: Any, transcript: str, needs_search: bool, source: str,
        score: float, reason: str,
    ) -> None:
        """Plain-English record of the one decision that shapes the whole turn:
        answer from the model's own knowledge, or go and look it up."""
        if source == "rules":
            how = "an instant keyword check (no model call needed)"
        elif source == "model":
            how = f"the router model, which scored this {score:.3f} on a 0-1 scale"
        else:
            how = "a failed router call, which defaults to searching to stay safe"
        lines = [
            "The assistant checked whether this question needs live information "
            "from the internet, or whether it already knows enough to answer.",
            f"How it decided: {how}.",
            f"Reason: {reason}.",
        ]
        if needs_search:
            lines.append(
                "Decision: this needs current information, so the assistant is "
                "searching the web before answering."
            )
        else:
            lines.append(
                "Decision: no search needed - the assistant will answer directly "
                "from its own knowledge. Nothing was added to its context."
            )
        self._write_detail(turn_id, "Deciding how to respond", lines)

    def narrate_web_search_start(self, turn_id: Any, query: str, provider: str, reason: str) -> None:
        self.web_search_count += 1
        self._write_detail(
            turn_id, f"Online search #{self.web_search_count} this session",
            [
                f"Why: {reason}",
                f'Search query: "{query}"',
                f"Search provider: {provider}",
            ],
        )

    def narrate_web_search_results(self, turn_id: Any, all_hits: list[dict], selected_hits: list[dict], max_results: int) -> None:
        if not all_hits:
            self._write_detail(turn_id, "Online search results", ["No results were returned by the web search."])
            return
        lines = [f"Found {len(all_hits)} result(s) from the web:"]
        for i, h in enumerate(all_hits, 1):
            score = float(h.get("similarity_score", 0.0))
            src = h.get("source", "unknown")
            text = str(h.get("text", ""))[:220]
            lines.append(f'{i}. relevance {score:.3f}, from "{src}": "{text}"')
        lines.append(
            f"Preprocessing: results were sorted by relevance (highest first) and the top "
            f"{min(len(selected_hits), max_results)} of {len(all_hits)} were kept for use."
        )
        self._write_detail(turn_id, "Online search results", lines)

    def narrate_summary(self, turn_id: Any, source: str, n_passages: int, summary_text: str, used_fallback: bool) -> None:
        method = (
            "a short extractive summary (the most relevant sentences picked out directly)"
            if used_fallback
            else "a small AI model that reads the passages and writes one short spoken sentence"
        )
        lines = [
            f"Why: to turn the {n_passages} result(s) from {source} into one short, "
            f"natural-sounding piece of information the assistant can work into its reply, "
            f"instead of reading raw web page text aloud.",
            f"How: {method}.",
            f'What it contains: "{summary_text}"' if summary_text else "Nothing usable was produced.",
        ]
        self._write_detail(turn_id, "Summary created", lines)

    def narrate_no_information(self, turn_id: Any) -> None:
        self._write_detail(
            turn_id, "No information found",
            [
                "The web search did not turn up anything relevant, so the assistant was "
                "told to answer from its own general knowledge instead of pretending to "
                "have a specific source."
            ],
        )

    def narrate_injection(
        self, turn_id: Any, injected_text: str, n_tokens_final: int, n_tokens_before_trim: int, max_tokens: int, kind: str
    ) -> None:
        if n_tokens_before_trim > max_tokens:
            preprocessing = (
                f"The information was shortened from {n_tokens_before_trim} to {max_tokens} tokens "
                f"(roughly word-pieces) so it stays short enough for the assistant to read quickly."
            )
        else:
            preprocessing = f"No shortening was needed ({n_tokens_final} tokens, under the {max_tokens}-token limit)."
        label = "grounding information" if kind == "ref" else "filler notice"
        lines = [
            f'What is given to the assistant: "{injected_text}"',
            "How it is given: it is fed to the assistant word-by-word as a hidden note "
            "inserted directly into the live, ongoing conversation -- the assistant's memory "
            "of everything said so far is NOT cleared or restarted, this is simply added on top of it.",
            f"Preprocessing: {preprocessing}",
            f"This {label} is exactly what the assistant receives as input before its next reply.",
        ]
        self._write_detail(turn_id, "Information given to the assistant", lines)

    def narrate_response(self, turn_id: Any, transcript: str, response_text: str) -> None:
        if response_text:
            self._write_detail(
                turn_id, "Assistant replied",
                [f'In reply to: "{transcript}"', f'The assistant said: "{response_text}"'],
            )
        else:
            self._write_detail(
                turn_id, "Assistant replied",
                [f'In reply to: "{transcript}"', "(no speech was captured for this turn)"],
            )

    def narrate_timing_summary(self, turn_id: Any, total_s: float, stages: dict[str, float]) -> None:
        ordered = sorted(stages.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"Total time from question to information ready: {total_s:.2f} seconds. Breakdown, slowest first:"]
        labels = {
            "quick_gate": "quick document check",
            "lookup_inject": "acknowledgement given to the assistant",
            "local_retrieval": "local document search",
            "web_search": "online search",
            "compression": "turning search results into a summary",
            "ref_inject": "giving the summary to the assistant",
        }
        for name, secs in ordered:
            pct = (secs / total_s * 100.0) if total_s > 0 else 0.0
            lines.append(f"  - {labels.get(name, name)}: {secs:.2f}s ({pct:.0f}% of the total)")
        self._write_detail(turn_id, "Timing breakdown", lines)

    def narrate_no_response_warning(self, turn_id: Any, transcript: str, elapsed_s: float) -> None:
        self._write_detail(
            turn_id, "No response detected",
            [
                f'In reply to: "{transcript}"',
                f"The assistant did not produce any spoken words in the {elapsed_s:.1f} seconds "
                "before the user spoke again. This is not normal -- some part of the pipeline "
                "produced nothing to say.",
            ],
        )

    def narrate_thinking_start(self, turn_id: Any) -> None:
        self._write_detail(
            turn_id, "Thinking sound started",
            ["Started playing so the user hears something while the assistant searches, instead of silence."],
        )

    def narrate_thinking_stop(self, turn_id: Any, reason: str, duration_s: float, play_count: int, clip_duration_s: float) -> None:
        if play_count <= 1:
            loop_desc = f"played once, did not need to loop (the clip is {clip_duration_s:.1f}s long)."
        else:
            loop_desc = f"played {play_count} times in a row (looped), the clip is {clip_duration_s:.1f}s long."
        self._write_detail(
            turn_id, "Thinking sound stopped",
            [
                f"Why it stopped: {reason}.",
                f"How long it played: {duration_s:.1f} seconds.",
                f"How many times it played: {loop_desc}",
            ],
        )
