#!/usr/bin/env python3
"""Copy MoreWaita SVG icons into dataset/{app-name}/icon.svg folders."""

import shutil
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MOREWAITA_DIR = REPO_ROOT / "vendor" / "MoreWaita" / "scalable" / "apps"
DATASET_DIR = REPO_ROOT / "dataset"


def icon_name_to_folder(filename: str) -> str:
    """Convert an SVG filename to a clean folder name.

    Examples:
        firefox.svg -> firefox
        org.gnome.Calculator.svg -> org.gnome.Calculator
        appimagekit-heroic.svg -> heroic
    """
    name = filename.removesuffix(".svg")
    # Strip appimagekit- prefix
    name = re.sub(r"^appimagekit-", "", name)
    return name


def collect():
    if not MOREWAITA_DIR.exists():
        print("MoreWaita submodule not found. Run: git submodule update --init")
        return

    src_icons = sorted(MOREWAITA_DIR.glob("*.svg"))
    # Only real files, not symlinks (symlinks are aliases for the same icon)
    real_icons = [p for p in src_icons if not p.is_symlink()]

    print(f"Found {len(real_icons)} unique icons in MoreWaita")

    copied = 0
    skipped = 0
    for svg_path in real_icons:
        folder_name = icon_name_to_folder(svg_path.name)
        dest_dir = DATASET_DIR / folder_name
        dest_svg = dest_dir / "icon.svg"

        if dest_svg.exists():
            skipped += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(svg_path, dest_svg)
        copied += 1

    print(f"Copied {copied} icons, skipped {skipped} (already exist)")


if __name__ == "__main__":
    collect()
