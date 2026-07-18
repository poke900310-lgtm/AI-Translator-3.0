from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def frame_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return -1


def add_label(image: Image.Image, label: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 22), fill=(255, 255, 255))
    draw.text((8, 4), label, fill=(0, 0, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build side-by-side debug composite sheets from capture/annotated/rendered frames."
    )
    ap.add_argument("debug_dir", help="Path to debug_artifacts directory")
    ap.add_argument("--limit", type=int, default=0, help="Max number of frames to export (0 = all)")
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    captures = debug_dir / "images" / "1.Capture"
    annotated = debug_dir / "images" / "2.Annotated"
    rendered = debug_dir / "images" / "3.Rendered"
    out_dir = debug_dir / "images" / "composites"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_files = sorted(captures.glob("frame_*.png"), key=frame_number)
    if args.limit > 0:
        cap_files = cap_files[: args.limit]

    for cap_path in cap_files:
        n = frame_number(cap_path)
        ann_path = annotated / cap_path.name
        ren_path = rendered / cap_path.name
        if not ren_path.exists():
            continue

        cap = add_label(Image.open(cap_path), f"capture {n}")
        ann = add_label(Image.open(ann_path), f"annotated {n}") if ann_path.exists() else None
        ren = add_label(Image.open(ren_path), f"rendered {n}")

        panels = [cap]
        if ann is not None:
            panels.append(ann)
        panels.append(ren)

        width = sum(p.width for p in panels)
        height = max(p.height for p in panels)
        sheet = Image.new("RGB", (width, height), (230, 230, 230))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width
        sheet.save(out_dir / cap_path.name)

    print(f"Wrote composites to: {out_dir}")


if __name__ == "__main__":
    main()
