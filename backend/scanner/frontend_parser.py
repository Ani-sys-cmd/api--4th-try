# backend/scanner/frontend_parser.py
"""
Frontend parser for detecting API calls in JS/TS/JSX/TSX files.

Exports:
  - parse_frontend_calls(root_path: str) -> List[Dict]
  - extract_calls_from_source(source: str, filename: str) -> List[Dict]

Each discovered call is represented as a dict:
{
  "type": "fetch" | "axios" | "xhr" | "jquery" | "websocket" | "other",
  "method": "GET|POST|PUT|DELETE|...|UNKNOWN",
  "url": "<raw url or template>",
  "file": "<filename>",
  "line": <int_line_number>,
  "context": "<nearby source snippet or comment>"
}
"""

from pathlib import Path
import re
from typing import List, Dict, Optional

# Simple regex patterns to capture common call sites.
FETCH_RE = re.compile(r"\bfetch\(\s*(`[^`]*`|\"[^\"]*\"|'[^']*')\s*(?:,)?", re.IGNORECASE)
AXIOS_RE = re.compile(r"\baxios\.(get|post|put|delete|patch|head|options)\s*\(\s*(`[^`]*`|\"[^\"]*\"|'[^']*')", re.IGNORECASE)
XHR_RE = re.compile(r"new\s+XMLHttpRequest\s*\(|\.open\(\s*(`[^`]*`|\"[^\"]*\"|'[^']*')\s*,\s*(`[^`]*`|\"[^\"]*\"|'[^']*')", re.IGNORECASE)
JQUERY_AJAX_RE = re.compile(r"\$\.\s*ajax\s*\(\s*({[\s\S]*?})\s*\)", re.IGNORECASE)
JQUERY_SHORT_RE = re.compile(r"\$\.(get|post)\s*\(\s*(`[^`]*`|\"[^\"]*\"|'[^']*')", re.IGNORECASE)
WS_SEND_RE = re.compile(r"\b(ws|socket)\.send\(\s*(`[^`]*`|\"[^\"]*\"|'[^']*')", re.IGNORECASE)

TEMPLATE_LITERAL_RE = re.compile(r"`([^`]+)`", re.IGNORECASE)
STRING_LITERAL_RE = re.compile(r"(['\"])(.*?)\1")

# heuristics for method from surrounding code if not explicitly provided
METHOD_IN_OPTIONS_RE = re.compile(r"\b(method|type)\s*:\s*['\"](GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)['\"]", re.IGNORECASE)


def _safe_read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return None


def _line_number_of_match(source: str, match_obj: re.Match) -> int:
    start = match_obj.start()
    return source.count("\n", 0, start) + 1


def extract_calls_from_source(source: str, filename: str) -> List[Dict]:
    calls: List[Dict] = []
    if not source:
        return calls

    # search for fetch(...)
    for m in FETCH_RE.finditer(source):
        raw = m.group(1).strip()
        # strip surrounding quotes/backticks
        url = raw.strip("`'\"")
        ln = _line_number_of_match(source, m)
        # try to detect method from subsequent options object
        method = "GET"
        # naive search for options object after the fetch(...) call (look ahead)
        lookahead = source[m.end():m.end()+400]
        mm = METHOD_IN_OPTIONS_RE.search(lookahead)
        if mm:
            method = mm.group(2).upper()
        # extract small context
        snippet = _extract_context(source, ln)
        calls.append({"type": "fetch", "method": method, "url": url, "file": filename, "line": ln, "context": snippet})

    # search for axios.<verb>(...)
    for m in AXIOS_RE.finditer(source):
        verb = m.group(1).upper()
        raw = m.group(2).strip()
        url = raw.strip("`'\"")
        ln = _line_number_of_match(source, m)
        snippet = _extract_context(source, ln)
        calls.append({"type": "axios", "method": verb, "url": url, "file": filename, "line": ln, "context": snippet})

    # search for $.get / $.post
    for m in JQUERY_SHORT_RE.finditer(source):
        verb = m.group(1).upper()
        raw = m.group(2).strip()
        url = raw.strip("`'\"")
        ln = _line_number_of_match(source, m)
        snippet = _extract_context(source, ln)
        calls.append({"type": "jquery", "method": verb, "url": url, "file": filename, "line": ln, "context": snippet})

    # search for XHR open(url,..)
    for m in XHR_RE.finditer(source):
        groups = m.groups()
        # groups may have url as second item
        url_raw = groups[-1] if groups else None
        if url_raw:
            url = url_raw.strip("`'\"")
        else:
            url = ""
        ln = _line_number_of_match(source, m)
        snippet = _extract_context(source, ln)
        calls.append({"type": "xhr", "method": "UNKNOWN", "url": url, "file": filename, "line": ln, "context": snippet})

    # websocket send
    for m in WS_SEND_RE.finditer(source):
        raw = m.group(2).strip()
        url = raw.strip("`'\"")
        ln = _line_number_of_match(source, m)
        snippet = _extract_context(source, ln)
        calls.append({"type": "websocket", "method": "SEND", "url": url, "file": filename, "line": ln, "context": snippet})

    return calls


def _extract_context(source: str, line_no: int, context_lines: int = 3) -> str:
    lines = source.splitlines()
    start = max(0, line_no - 1 - context_lines)
    end = min(len(lines), line_no - 1 + context_lines + 1)
    snippet = "\n".join(lines[start:end])
    # also include any immediately preceding comment block (/** ... */) or single-line comment
    # scan backward for comments up to 6 lines
    comment_lines = []
    for i in range(max(0, start-6), start):
        l = lines[i].strip()
        if l.startswith("//") or l.startswith("/*") or l.startswith("*"):
            comment_lines.append(lines[i])
        else:
            # break when encountering non-comment non-empty line
            if l:
                break
    if comment_lines:
        return "\n".join(comment_lines + [snippet])
    return snippet


def _is_js_like(path: Path) -> bool:
    return path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}


def parse_frontend_calls(root_path: str) -> List[Dict]:
    """
    Walk the root_path and parse frontend files for API calls.
    Returns a list of call dicts.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root_path}")

    results: List[Dict] = []
    # Prefer scanning src/ and public/ directories first for typical React projects
    prioritized = ["src", "public", "app", "client"]
    # Collect files to parse
    files_to_scan = []
    for p in root.rglob("*"):
        if p.is_file() and _is_js_like(p):
            files_to_scan.append(p)

    # Heuristic: sort files to scan so that prioritized directories come first
    def _priority_key(p: Path):
        for i, name in enumerate(prioritized):
            if name in p.parts:
                return i
        return len(prioritized)

    files_to_scan = sorted(files_to_scan, key=_priority_key)

    for f in files_to_scan:
        src = _safe_read(f)
        if not src:
            continue
        try:
            calls = extract_calls_from_source(src, str(f))
            if calls:
                results.extend(calls)
        except Exception:
            # non-fatal; continue
            continue

    # Optional post-processing: normalize relative URLs, strip query fragments for grouping
    for call in results:
        call["url_normalized"] = _normalize_url(call["url"])

    return results


def _normalize_url(url: str) -> str:
    if not url:
        return url
    # strip template expressions like ${...} (leave placeholder)
    url = re.sub(r"\$\{[^}]+\}", "{param}", url)
    # remove query string for normalized grouping
    if "?" in url:
        url = url.split("?", 1)[0]
    return url
