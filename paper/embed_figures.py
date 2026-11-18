"""Produce a self-contained, shareable copy of the paper draft.

Reads a source markdown draft whose images are ordinary relative-path links
(e.g. ../analysis/destiny2/outputs/figures/05a_overall_xcorr.png), replaces
each with a base64 data URI, and writes <stem>_standalone.md next to it.
The standalone file renders images in viewers that support data URIs
(VSCode preview, Typora, Obsidian); GitHub's web renderer does not.

Usage: python embed_figures.py [draft_v1.md]
"""

import base64
import re
import sys
from pathlib import Path

IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(((?!data:)[^)]+?\.png)\)')


def embed(src: Path) -> Path:
    text = src.read_text(encoding='utf-8')
    root = src.parent
    missing = []

    def repl(match):
        alt, rel = match.group(1), match.group(2)
        img = (root / rel).resolve()
        if not img.exists():
            missing.append(rel)
            return match.group(0)
        b64 = base64.b64encode(img.read_bytes()).decode('ascii')
        print(f'embedded {img.name} ({img.stat().st_size // 1024} KB)')
        return f'![{alt}](data:image/png;base64,{b64})'

    out_text = IMG_PATTERN.sub(repl, text)
    out = src.with_name(src.stem + '_standalone' + src.suffix)
    out.write_text(out_text, encoding='utf-8')
    print(f'wrote {out.name} ({out.stat().st_size // 1024} KB)')
    if missing:
        print(f'WARNING - {len(missing)} image(s) not found: {missing}')
    return out


if __name__ == '__main__':
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'draft_v1.md'
    embed(source)
