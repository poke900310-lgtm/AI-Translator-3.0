from __future__ import annotations

import requests

# Bump PROMPT_VERSION when ``_build_system_prompt`` changes in a way that
# should auto-invalidate cached translations. Bump SCHEMA_VERSION when the
# on-disk translation-memory file shape changes. Both feed
# ``cache_fingerprint`` which the controller writes alongside every cached
# entry; on load, entries whose fingerprint doesn't match the current build
# are dropped so a prompt edit can never silently keep returning stale
# translations.
PROMPT_VERSION = "v3"
SCHEMA_VERSION = "v2"


def cache_fingerprint(source_hint: str | None, model: str | None) -> str:
    """Cache-key fingerprint covering everything that can change a
    translation's correctness for the same source text: the prompt template,
    the on-disk file shape, the model id, and the source-language hint.
    """
    return "::".join(
        [
            PROMPT_VERSION,
            SCHEMA_VERSION,
            (model or "local-model").strip() or "local-model",
            (source_hint or "").strip() or "auto",
        ]
    )


class LlamaServerEndpointError(RuntimeError):
    pass


# Use a session that ignores proxy env vars (common corporate setup breaks localhost calls)
_SESSION = requests.Session()
_SESSION.trust_env = False


def _normalize_base(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:8080"
    if "://" not in base:
        base = "http://" + base
    return base.rstrip("/")


def _candidate_bases(base_url: str) -> list[str]:
    base = _normalize_base(base_url)
    cands = [base]

    # helpful localhost/ipv4 swap
    if "localhost" in base:
        cands.append(base.replace("localhost", "127.0.0.1"))
    if "127.0.0.1" in base:
        cands.append(base.replace("127.0.0.1", "localhost"))

    # de-dup preserve order
    out, seen = [], set()
    for b in cands:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _probe_base(base_url: str, timeout_s: int) -> dict:
    base = _normalize_base(base_url)
    probes = [
        ("/health", "GET"),
        ("/v1/health", "GET"),
        ("/v1/models", "GET"),
        ("/", "GET"),
    ]
    out: dict = {}
    for path, method in probes:
        url = base + path
        try:
            r = _SESSION.request(method, url, timeout=min(10, timeout_s))
            out[path] = {"status": r.status_code, "content_type": r.headers.get("content-type", "")}
        except Exception as e:
            out[path] = {"error": repr(e)}
    return out


def probe_endpoints(base_url: str, timeout_s: int = 10) -> str:
    info = _probe_base(base_url, timeout_s)
    parts = []
    for path, d in info.items():
        if "status" in d:
            parts.append(f"{path} -> {d['status']} {d.get('content_type','')}".strip())
        else:
            parts.append(f"{path} -> {d['error']}")
    return "; ".join(parts)


def _build_system_prompt(
    source_hint: str | None, *, with_context: bool, context_kind: str = ""
) -> str:
    src = source_hint or "the source language"
    lines = [
        "You are a translation engine.",
        f"Task: Translate from {src} to natural English.",
        "Rules:",
        "- Output ONLY the translated text.",
        "- Do NOT add explanations, notes, headings, or markdown.",
        "- Preserve line breaks where possible; do not add extra blank lines.",
    ]
    if with_context:
        lines.append(
            "- The CONTEXT section shows recent preceding lines from the same"
            " scene, with their translations. Use them to resolve pronouns,"
            " dropped subjects, tone, speaker voice, and callbacks. Translate"
            " ONLY the INPUT — never re-emit the context lines."
        )
    kind = (context_kind or "").lower()
    if kind == "ui":
        # Menu labels come in isolation (no sentence context), so the LLM
        # was picking the wrong reading for words like ロード, セーブ, etc.
        # Anchoring on standard game-UI conventions cuts those wrong picks.
        lines.append(
            "- The INPUT is a game UI element — a menu label, button, HUD"
            " text, or config option. Use the standard localized game-UI"
            " reading, not a literal one. Examples: ロード → Load;"
            " クイックロード → Quick Load; セーブ → Save; クイックセーブ →"
            " Quick Save; コンフィグ → Config; バックログ → Backlog; オプション"
            " → Options; タイトルに戻る → Return to Title; ゲーム終了 → Quit"
            " Game; 戻る → Back; はい → Yes; いいえ → No."
        )
    elif kind == "name":
        lines.append(
            "- The INPUT is a character-name label (speaker tag next to a"
            " dialogue line). Output the name only; do not translate honorifics"
            " into English titles."
        )
    return "\n".join(lines) + "\n"


def _build_user_block(text: str, history: list[tuple[str, str]] | None) -> str:
    parts: list[str] = []
    if history:
        parts.append("CONTEXT (recent lines, oldest first):")
        for src, tgt in history:
            src_clean = (src or "").strip()
            tgt_clean = (tgt or "").strip()
            if not src_clean or not tgt_clean:
                continue
            # Keep each pair on its own block so multi-line dialogue stays
            # legible in the prompt without a delimiter colliding with the
            # source text (which can contain arbitrary punctuation).
            parts.append("SOURCE:\n" + src_clean + "\nTRANSLATION:\n" + tgt_clean)
        parts.append("")
    parts.append("INPUT:\n" + text + "\n\nOUTPUT:\n")
    return "\n".join(parts)


def translate_llama_server(
    text: str,
    source_hint: str | None,
    base_url: str,
    model: str = "local-model",
    timeout_s: int = 60,
    max_tokens: int = 512,
    history: list[tuple[str, str]] | None = None,
    context_kind: str = "",
) -> str:
    """
    Translate via llama.cpp llama-server.

    Tries OpenAI-compatible /v1/chat/completions first, then falls back to /completion.

    ``history`` is an ordered list of ``(source, translation)`` pairs from
    recently-translated lines in the same scene, oldest-first. Passing a
    non-empty history makes the prompt include a CONTEXT section that the
    model uses to resolve pronouns, dropped subjects, and tone against
    what was just said. Pass ``None`` or an empty list to translate the
    line in isolation (legacy behaviour).

    ``context_kind`` tags the input as ``"ui"`` (game menu label / button),
    ``"name"`` (speaker tag), or ``""`` (no special class). Adds a rule to
    the system prompt anchoring the LLM on the correct reading — e.g.
    stops クイックロード from becoming "Quick Road" when translated in
    isolation.
    """
    if not text:
        return ""

    has_context = bool(history)
    system = _build_system_prompt(source_hint, with_context=has_context, context_kind=context_kind)
    user = _build_user_block(text, history)

    last_err: Exception | None = None

    for base in _candidate_bases(base_url):
        # 1) OpenAI chat completions
        try:
            url = base + "/v1/chat/completions"
            payload = {
                "model": model or "local-model",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "stream": False,
                "max_tokens": int(max_tokens),
            }
            r = _SESSION.post(url, json=payload, timeout=timeout_s)
            if r.status_code != 404:
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    out = (msg.get("content") or "").strip()
                    if out:
                        return out
        except Exception as e:
            last_err = e

        # 2) Native /completion
        try:
            url = base + "/completion"
            # for /completion we include the full instruction in the prompt
            prompt = system + "\n" + user
            payload = {
                "prompt": prompt,
                "temperature": 0,
                "stream": False,
                "n_predict": int(max_tokens),
            }
            r = _SESSION.post(url, json=payload, timeout=timeout_s)
            if r.status_code != 404:
                r.raise_for_status()
                data = r.json()
                out = (data.get("content") or "").strip()
                if out:
                    return out
        except Exception as e:
            last_err = e

    raise LlamaServerEndpointError(
        f"Could not reach llama-server at {base_url!r}. "
        f"Last error: {last_err!r}. "
        f"Probe: {probe_endpoints(base_url, timeout_s=min(10, timeout_s))}"
    )
