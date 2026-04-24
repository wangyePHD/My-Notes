#!/usr/bin/env python3
"""Convert arXiv reading Markdown notes to the local LaTeX note style."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "reading" / "arxiv_weekly"

SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "\ufe0f"
    "]+"
)


def strip_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text).strip()


def escape_text(text: str) -> str:
    return "".join(SPECIALS.get(ch, ch) for ch in text)


def escape_code(text: str) -> str:
    return escape_text(text).replace(" ", r"\ ")


def split_math_segments(text: str) -> list[tuple[bool, str]]:
    parts: list[tuple[bool, str]] = []
    pos = 0
    for match in re.finditer(r"\$[^$\n]+\$", text):
        if match.start() > pos:
            parts.append((False, text[pos : match.start()]))
        parts.append((True, match.group(0)))
        pos = match.end()
    if pos < len(text):
        parts.append((False, text[pos:]))
    return parts


TOKEN_PATTERN = re.compile(
    r"(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\([^)]+\))|(\*[^*\n]+\*)|(<https?://[^>]+>)"
)


def convert_plain_segment(text: str) -> str:
    text = strip_emoji(text)
    chunks: list[str] = []
    pos = 0
    for match in TOKEN_PATTERN.finditer(text):
        if match.start() > pos:
            chunks.append(escape_text(text[pos : match.start()]))

        token = match.group(0)
        if token.startswith("`"):
            chunks.append(rf"\code{{{escape_code(token[1:-1])}}}")
        elif token.startswith("**"):
            chunks.append(rf"\textbf{{{convert_inline(token[2:-2])}}}")
        elif token.startswith("["):
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            if link:
                chunks.append(rf"\href{{{link.group(2)}}}{{{convert_inline(link.group(1))}}}")
            else:
                chunks.append(escape_text(token))
        elif token.startswith("*"):
            chunks.append(rf"\textit{{{convert_inline(token[1:-1])}}}")
        elif token.startswith("<"):
            chunks.append(rf"\url{{{token[1:-1]}}}")
        else:
            chunks.append(escape_text(token))
        pos = match.end()

    if pos < len(text):
        chunks.append(escape_text(text[pos:]))
    return "".join(chunks)


def convert_inline(text: str, parse_bold_spans: bool = True) -> str:
    if parse_bold_spans and "**" in text:
        chunks: list[str] = []
        pos = 0
        for match in re.finditer(r"\*\*(.+?)\*\*", text):
            if match.start() > pos:
                chunks.append(convert_inline(text[pos : match.start()], parse_bold_spans=False))
            chunks.append(rf"\textbf{{{convert_inline(match.group(1), parse_bold_spans=False)}}}")
            pos = match.end()
        if pos < len(text):
            chunks.append(convert_inline(text[pos:], parse_bold_spans=False))
        return "".join(chunks)

    chunks: list[str] = []
    for is_math, segment in split_math_segments(text):
        if is_math:
            chunks.append(segment)
        else:
            chunks.append(convert_plain_segment(segment))
    return "".join(chunks)


def parse_table(lines: list[str], start: int) -> tuple[list[str], int]:
    table_lines = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1
    return table_lines, i


def is_table_separator(row: str) -> bool:
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def convert_table(table_lines: list[str]) -> list[str]:
    rows = [line.strip("|").split("|") for line in table_lines]
    rows = [[cell.strip() for cell in row] for row in rows if not is_table_separator("|".join(row))]
    if not rows:
        return []

    col_count = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (col_count - len(row)))

    if col_count == 2:
        spec = r"p{3.2cm}p{10.5cm}"
    elif col_count == 3:
        spec = r"p{2.8cm}p{4.8cm}p{6.2cm}"
    elif col_count == 4:
        spec = r"p{0.9cm}p{3.0cm}p{3.0cm}p{6.0cm}"
    else:
        spec = r"p{0.7cm}p{2.5cm}p{2.6cm}p{2.3cm}p{5.0cm}"

    out = [r"\small", rf"\begin{{longtable}}{{{spec}}}", r"\toprule"]
    for index, row in enumerate(rows):
        converted = [convert_inline(cell) for cell in row[:col_count]]
        out.append(" & ".join(converted) + r" \\")
        if index == 0:
            out.append(r"\midrule")
    out.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return out


def heading_command(level: int, text: str) -> str:
    if level == 1:
        command = "section"
    elif level == 2:
        command = "subsection"
    elif level == 3:
        command = "subsubsection"
    else:
        command = "paragraph"
    return rf"\{command}{{{convert_inline(text)}}}"


def close_list(out: list[str], active_list: str | None) -> str | None:
    if active_list:
        out.append(rf"\end{{{active_list}}}")
        out.append("")
    return None


def flush_quote(out: list[str], quote_lines: list[str]) -> None:
    if not quote_lines:
        return
    out.append(r"\begin{conceptbox}")
    for index, line in enumerate(quote_lines):
        suffix = r"\\" if index < len(quote_lines) - 1 else ""
        out.append(convert_inline(line.strip().rstrip("  ")) + suffix)
    out.append(r"\end{conceptbox}")
    out.append("")
    quote_lines.clear()


def convert_file(path: Path, source_text: str | None = None) -> Path:
    if source_text is None:
        source_text = path.read_text(encoding="utf-8")
    lines = source_text.splitlines()
    out: list[str] = []
    active_list: str | None = None
    quote_lines: list[str] = []
    in_display_math = False
    skip_toc = False
    i = 0

    out.append("% ============================================================")
    out.append(f"% Converted from {path.name}")
    out.append("% ============================================================")
    out.append("")

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped == "---":
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            i += 1
            continue

        if stripped == "$$":
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            out.append(r"\]" if in_display_math else r"\[")
            in_display_math = not in_display_math
            i += 1
            continue

        if stripped == r"\[":
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            out.append(raw)
            in_display_math = True
            i += 1
            continue

        if stripped == r"\]":
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            out.append(raw)
            in_display_math = False
            i += 1
            continue

        if in_display_math:
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            out.append(raw)
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 2 and text == "目录":
                skip_toc = True
                i += 1
                continue
            skip_toc = False
            out.append(heading_command(level, text))
            out.append("")
            i += 1
            continue

        if skip_toc:
            i += 1
            continue

        if stripped.startswith(">"):
            active_list = close_list(out, active_list)
            quote_lines.append(stripped[1:].strip())
            i += 1
            continue

        if stripped.startswith("|"):
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            table_lines, next_i = parse_table(lines, i)
            out.extend(convert_table(table_lines))
            out.append("")
            i = next_i
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if bullet or ordered:
            flush_quote(out, quote_lines)
            env = "itemize" if bullet else "enumerate"
            if active_list != env:
                active_list = close_list(out, active_list)
                active_list = env
                out.append(rf"\begin{{{env}}}[leftmargin=1.5em]")
            item_text = bullet.group(1) if bullet else ordered.group(1)
            out.append(rf"  \item {convert_inline(item_text)}")
            i += 1
            continue

        if not stripped:
            active_list = close_list(out, active_list)
            flush_quote(out, quote_lines)
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue

        active_list = close_list(out, active_list)
        flush_quote(out, quote_lines)
        out.append(convert_inline(stripped))
        out.append("")
        i += 1

    close_list(out, active_list)
    flush_quote(out, quote_lines)

    tex_path = path.with_suffix(".tex")
    tex_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    return tex_path


def git_markdown_sources() -> list[tuple[Path, str]]:
    listed = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "reading/arxiv_weekly"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    sources: list[tuple[Path, str]] = []
    for rel in listed:
        if not rel.endswith(".md"):
            continue
        text = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, text=True)
        sources.append((ROOT / rel, text))
    return sources


def main() -> None:
    disk_sources = [(path, None) for path in sorted(SOURCE_DIR.glob("*.md"))]
    sources = disk_sources or git_markdown_sources()
    for path, source_text in sources:
        tex_path = convert_file(path, source_text)
        if path.exists():
            path.unlink()
        print(f"{path.name} -> {tex_path.name}")


if __name__ == "__main__":
    main()
