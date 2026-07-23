"""Fetch and cache the real-world training datasets to ml/data/.

Run with:

    python ml/download_datasets.py            # fetch everything missing
    python ml/download_datasets.py --force    # re-fetch even if cached

Sources (URLs verified reachable at authoring time, 2026-07-22):

* NASA C-MAPSS turbofan degradation — run-to-failure with RUL ground truth.
  NASA PCoE mirror on S3.
* UCI AI4I 2020 Predictive Maintenance — 10k rows with labelled failure modes.
  UCI ML Repository static endpoint.
* CWRU bearing vibration — accelerometer captures of healthy and seeded-fault
  bearings, from the Case Western Bearing Data Center.

POLICY: if a source is unreachable this script FAILS LOUDLY and prints the
manual download instructions. It never substitutes generated data — a model
trained on data whose rules we wrote ourselves would be circular and its RUL
numbers meaningless.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
USER_AGENT = "FactoryPilot-AI/0.1 (dataset fetch for predictive maintenance research)"
TIMEOUT_S = 120


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str
    min_bytes: int  # sanity floor: an HTML error page must not pass as data


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    files: tuple[RemoteFile, ...]
    manual_instructions: str
    #: Zip members to extract, if the payload is an archive. Empty = not a zip.
    unzip_members_prefix: tuple[str, ...] = field(default_factory=tuple)


CMAPSS = Dataset(
    key="cmapss",
    title="NASA C-MAPSS turbofan degradation (RUL ground truth)",
    files=(
        RemoteFile(
            url=(
                "https://phm-datasets.s3.amazonaws.com/NASA/"
                "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
            ),
            filename="CMAPSS.zip",
            min_bytes=5_000_000,
        ),
    ),
    # The S3 zip nests CMaps.zip / text files depending on packaging vintage;
    # extraction below handles both layouts.
    unzip_members_prefix=("train_FD", "test_FD", "RUL_FD"),
    manual_instructions=(
        "Manual fallback: search 'NASA PCoE Turbofan Engine Degradation "
        "Simulation Data Set' (data.nasa.gov or the PCoE data repository), "
        "download the C-MAPSS archive, and place train_FD001.txt, "
        "test_FD001.txt and RUL_FD001.txt in ml/data/cmapss/."
    ),
)

AI4I = Dataset(
    key="ai4i2020",
    title="UCI AI4I 2020 Predictive Maintenance (labelled failure modes)",
    files=(
        RemoteFile(
            url=(
                "https://archive.ics.uci.edu/static/public/601/"
                "ai4i+2020+predictive+maintenance+dataset.zip"
            ),
            filename="ai4i2020.zip",
            min_bytes=100_000,
        ),
    ),
    unzip_members_prefix=("ai4i2020",),
    manual_instructions=(
        "Manual fallback: https://archive.ics.uci.edu/dataset/601 — download "
        "the dataset zip and place ai4i2020.csv in ml/data/ai4i2020/."
    ),
)

#: CWRU drive-end 12kHz captures at 1797 rpm / 0 HP load:
#: 97 = healthy baseline, 105 = inner race 0.007", 118 = ball 0.007",
#: 130 = outer race 0.007" @6:00. Enough to validate fault-signature features.
CWRU = Dataset(
    key="cwru",
    title="CWRU bearing vibration (bearing fault signatures)",
    files=(
        RemoteFile("https://engineering.case.edu/sites/default/files/97.mat", "97_normal.mat", 100_000),
        RemoteFile("https://engineering.case.edu/sites/default/files/105.mat", "105_inner_race.mat", 100_000),
        RemoteFile("https://engineering.case.edu/sites/default/files/118.mat", "118_ball.mat", 100_000),
        RemoteFile("https://engineering.case.edu/sites/default/files/130.mat", "130_outer_race.mat", 100_000),
    ),
    manual_instructions=(
        "Manual fallback: https://engineering.case.edu/bearingdatacenter — "
        "download 97.mat, 105.mat, 118.mat, 130.mat (12k drive end, 0.007\", "
        "0 HP) and place them in ml/data/cwru/ under the names "
        "97_normal.mat, 105_inner_race.mat, 118_ball.mat, 130_outer_race.mat."
    ),
)

DATASETS = (CMAPSS, AI4I, CWRU)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _fetch(remote: RemoteFile, dest: Path) -> None:
    """Stream one URL to ``dest``, atomically, with a size sanity check."""
    request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                shutil.copyfileobj(response, tmp)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    size = tmp_path.stat().st_size
    if size < remote.min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise IOError(
            f"Downloaded only {size} bytes from {remote.url} "
            f"(expected >= {remote.min_bytes}). Likely an error page, not data."
        )
    shutil.move(str(tmp_path), str(dest))


def _extract_zip(archive: Path, out_dir: Path, member_prefixes: tuple[str, ...]) -> list[Path]:
    """Extract wanted members, flattening paths and recursing into nested zips."""
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if not base or info.is_dir():
                continue
            if base.lower().endswith(".zip"):
                # C-MAPSS ships as a zip inside a zip in some packagings.
                nested = out_dir / base
                nested.write_bytes(zf.read(info))
                extracted.extend(_extract_zip(nested, out_dir, member_prefixes))
                nested.unlink()
            elif any(base.startswith(p) for p in member_prefixes):
                target = out_dir / base
                target.write_bytes(zf.read(info))
                extracted.append(target)
    return extracted


def _download_dataset(dataset: Dataset, force: bool) -> list[Path]:
    out_dir = DATA_DIR / dataset.key
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    for remote in dataset.files:
        dest = out_dir / remote.filename
        if dest.exists() and not force:
            print(f"  cached   {dest.relative_to(DATA_DIR.parent)}")
        else:
            print(f"  fetching {remote.url}")
            try:
                _fetch(remote, dest)
            except (urllib.error.URLError, IOError, OSError) as exc:
                print(
                    f"\nFAILED to download '{dataset.title}':\n  {exc}\n\n"
                    f"{dataset.manual_instructions}\n\n"
                    "This script will NOT substitute generated data — a model "
                    "trained on data whose rules we wrote is circular.",
                    file=sys.stderr,
                )
                raise SystemExit(1) from exc
            print(f"  saved    {dest.relative_to(DATA_DIR.parent)} ({dest.stat().st_size:,} bytes)")
        produced.append(dest)

        if dataset.unzip_members_prefix and dest.suffix == ".zip":
            members = _extract_zip(dest, out_dir, dataset.unzip_members_prefix)
            if not members and not any(
                p.name.startswith(dataset.unzip_members_prefix) for p in out_dir.iterdir()
            ):
                print(
                    f"\nArchive {dest.name} contained none of the expected members "
                    f"{dataset.unzip_members_prefix}.\n{dataset.manual_instructions}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            for m in members:
                print(f"  extract  {m.relative_to(DATA_DIR.parent)}")

    return produced


# ---------------------------------------------------------------------------
# Summaries — prove what arrived is what training expects
# ---------------------------------------------------------------------------
def _summarize_cmapss() -> None:
    import pandas as pd

    path = DATA_DIR / "cmapss" / "train_FD001.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    engines = df[0].nunique()
    print(
        f"  C-MAPSS  train_FD001: shape={df.shape} "
        f"({engines} engines, {df.shape[1] - 5} sensor channels, run-to-failure)"
    )


def _summarize_ai4i() -> None:
    import pandas as pd

    path = DATA_DIR / "ai4i2020" / "ai4i2020.csv"
    df = pd.read_csv(path)
    failures = int(df["Machine failure"].sum())
    print(
        f"  AI4I2020 shape={df.shape}, failures={failures} "
        f"({failures / len(df):.1%} — the imbalance the classifier must handle)"
    )


def _summarize_cwru() -> None:
    from scipy.io import loadmat

    for name in ("97_normal", "105_inner_race", "118_ball", "130_outer_race"):
        mat = loadmat(DATA_DIR / "cwru" / f"{name}.mat")
        de_keys = [k for k in mat if k.endswith("DE_time")]
        if not de_keys:
            raise SystemExit(
                f"CWRU file {name}.mat has no drive-end accelerometer channel "
                f"(keys: {[k for k in mat if not k.startswith('__')]}). "
                "The download is not the expected capture."
            )
        signal = mat[de_keys[0]].ravel()
        print(f"  CWRU     {name}: {signal.size:,} samples @12kHz drive-end")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        print(f"\n{dataset.title}")
        _download_dataset(dataset, force=args.force)

    print("\nSummary")
    print("=" * 60)
    _summarize_cmapss()
    _summarize_ai4i()
    _summarize_cwru()
    print("=" * 60)
    print("All datasets present. Next: python ml/train_rul.py")


if __name__ == "__main__":
    main()
