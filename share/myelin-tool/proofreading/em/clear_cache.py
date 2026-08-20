"""Reclaim tube-cache disk space.

    uv run python -m proofreading.em.clear_cache                    # list what's there
    uv run python -m proofreading.em.clear_cache --stale-masks      # DRY RUN
    uv run python -m proofreading.em.clear_cache --stale-masks -f   # actually delete
    uv run python -m proofreading.em.clear_cache --cell 8646911355725309811 -f
    uv run python -m proofreading.em.clear_cache --all -f

The tube cache under ``<wal_dir>/tube_cache`` is **derived** data -- EM/mask chunks fetched from
GCS, plus the per-branch ``.done`` markers -- so deleting it only costs re-download time. The
``*.jsonl`` **WAL logs** in ``<wal_dir>`` itself are the irreplaceable record of every annotation
and are NEVER touched here: this tool refuses to operate outside ``tube_cache`` (see
:func:`_assert_inside_tube_cache`), and every mode is a dry run until you pass ``-f``.

Modes
-----
``--stale-masks``  Delete only mask volumes whose mip is NOT the one currently in effect
                   (``PROOFREAD_TGT_MIP``, default :attr:`CellTube.DEFAULT_TGT_MIP`), plus their
                   now-meaningless markers. Keeps all EM chunks, so nothing re-downloads unless
                   you switch the mask mip back. This is the cheap win after changing the default.
``--cell ROOT``    Delete one cell's whole tube cache (EM + every mask + markers).
``--all``          Delete every cell's tube cache.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

from .tube import CellTube

_MASK_DIR_RE = re.compile(r"^tgt(?:_mip(\d+))?$")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def default_wal_dir() -> Path:
    env = os.environ.get("PROOFREAD_WAL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "proofread_sessions"


def _assert_inside_tube_cache(target: Path, tube_cache: Path) -> None:
    """Refuse to delete anything that isn't strictly inside ``tube_cache``.

    The WAL logs live one level up in ``wal_dir``; a path bug here would destroy real annotation
    data, so this is a hard gate rather than a convention.
    """
    t, root = target.resolve(), tube_cache.resolve()
    if t == root or root not in t.parents:
        raise SystemExit(f"refusing to delete outside tube_cache: {t}")
    if any(p.suffix == ".jsonl" for p in [t, *t.rglob("*.jsonl")] if p.is_file()):
        raise SystemExit(f"refusing: {t} contains WAL .jsonl files")


def _mask_mip(dirname: str, em_mip: int) -> int | None:
    """The mask mip a mask directory holds, or None if it isn't a mask dir.

    Bare ``tgt`` predates configurable mask mips, when the mask was always built at the EM mip.
    """
    m = _MASK_DIR_RE.match(dirname)
    if not m:
        return None
    return em_mip if m.group(1) is None else int(m.group(1))


def iter_cells(tube_cache: Path):
    for ds in sorted(p for p in tube_cache.iterdir() if p.is_dir()):
        for cell in sorted(p for p in ds.iterdir() if p.is_dir()):
            yield ds.name, cell


def cmd_list(tube_cache: Path, em_mip: int, tgt_mip: int) -> None:
    total = 0
    print(f"tube_cache: {tube_cache}")
    print(f"mask mip in effect: {tgt_mip}  (EM mip {em_mip})\n")
    for ds, cell in iter_cells(tube_cache):
        size = _dir_size(cell)
        total += size
        print(f"  {_human(size):>9}  {ds}/{cell.name}")
        for sub in sorted(p for p in cell.iterdir() if p.is_dir()):
            mm = _mask_mip(sub.name, em_mip)
            tag = ""
            if mm is not None:
                tag = "  <- IN USE" if mm == tgt_mip else "  <- STALE (--stale-masks)"
            print(f"  {_human(_dir_size(sub)):>9}    {sub.name}{tag}")
    print(f"\n  {_human(total):>9}  TOTAL")


def _remove(target: Path, tube_cache: Path, force: bool) -> int:
    size = _dir_size(target) if target.is_dir() else target.stat().st_size
    _assert_inside_tube_cache(target, tube_cache)
    if force:
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    return size


def cmd_stale_masks(tube_cache: Path, em_mip: int, tgt_mip: int, force: bool) -> int:
    freed = 0
    for ds, cell in iter_cells(tube_cache):
        for sub in sorted(p for p in cell.iterdir() if p.is_dir()):
            mm = _mask_mip(sub.name, em_mip)
            if mm is None or mm == tgt_mip:
                continue
            n = _remove(sub, tube_cache, force)
            freed += n
            print(f"  {'removed' if force else 'would remove'} {_human(n):>9}  {ds}/{cell.name}/{sub.name}")
            # the mask markers for that mip are meaningless once its chunks are gone
            markers = cell / "_branches"
            if markers.is_dir():
                pat = f".tgt{mm}.done"
                stale = [m for m in markers.iterdir() if m.name.endswith(pat)]
                # bare `tgt` was covered by the legacy `path_N.done`, which ALSO vouches for em --
                # so leave those alone; dropping them would force a needless EM refetch.
                for m in stale:
                    freed += _remove(m, tube_cache, force)
                if stale:
                    print(f"            {'removed' if force else 'would remove'} {len(stale)} stale {pat} markers")
    return freed


def cmd_cell(tube_cache: Path, root_id: str, force: bool) -> int:
    hits = [(ds, c) for ds, c in iter_cells(tube_cache) if c.name == str(root_id)]
    if not hits:
        print(f"no tube cache for cell {root_id}")
        return 0
    freed = 0
    for ds, cell in hits:
        n = _remove(cell, tube_cache, force)
        freed += n
        print(f"  {'removed' if force else 'would remove'} {_human(n):>9}  {ds}/{cell.name}")
    return freed


def cmd_all(tube_cache: Path, force: bool) -> int:
    freed = 0
    for ds, cell in iter_cells(tube_cache):
        n = _remove(cell, tube_cache, force)
        freed += n
        print(f"  {'removed' if force else 'would remove'} {_human(n):>9}  {ds}/{cell.name}")
    return freed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wal-dir", default=None, help="default: $PROOFREAD_WAL_DIR or <repo>/proofread_sessions")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stale-masks", action="store_true", help="delete mask volumes not at the mip in effect")
    g.add_argument("--cell", metavar="ROOT_ID", help="delete one cell's whole tube cache")
    g.add_argument("--all", action="store_true", help="delete every cell's tube cache")
    ap.add_argument("-f", "--force", action="store_true", help="actually delete (default is a dry run)")
    args = ap.parse_args(argv)

    wal_dir = Path(args.wal_dir) if args.wal_dir else default_wal_dir()
    tube_cache = wal_dir / "tube_cache"
    if not tube_cache.is_dir():
        print(f"no tube_cache at {tube_cache}")
        return 0

    em_mip = 1  # matches CellReviewService's tube_mip default
    tgt_mip = CellTube.DEFAULT_TGT_MIP

    if not (args.stale_masks or args.cell or args.all):
        cmd_list(tube_cache, em_mip, tgt_mip)
        return 0

    if not args.force:
        print("DRY RUN -- nothing deleted. Re-run with -f to apply.\n")
    if args.stale_masks:
        freed = cmd_stale_masks(tube_cache, em_mip, tgt_mip, args.force)
    elif args.cell:
        freed = cmd_cell(tube_cache, args.cell, args.force)
    else:
        freed = cmd_all(tube_cache, args.force)
    print(f"\n{_human(freed)} {'freed' if args.force else 'would be freed'}")
    if not args.force:
        print("(dry run -- re-run with -f)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
