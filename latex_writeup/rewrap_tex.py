from pathlib import Path
import textwrap
import re
import sys

WIDTH = int(sys.argv[2])
FOLDER = Path(sys.argv[1])

def rewrap(text):
    lines = text.splitlines()
    out = []
    para = []

    def flush():
        nonlocal para
        if not para:
            return
        joined = " ".join(x.strip() for x in para)
        wrapped = textwrap.fill(
            joined,
            width=WIDTH,
            break_long_words=False,
            break_on_hyphens=False
        )
        out.extend(wrapped.splitlines())
        para = []

    for line in lines:
        s = line.strip()

        if not s:
            flush()
            out.append("")
            continue

        if s.startswith("\\") or s.startswith("%"):
            flush()
            out.append(line)
            continue

        para.append(line)

    flush()
    return "\n".join(out) + "\n"

for file in FOLDER.rglob("*.tex"):
    old = file.read_text(encoding="utf8")
    new = rewrap(old)
    file.write_text(new, encoding="utf8")
    print("rewrapped", file)
