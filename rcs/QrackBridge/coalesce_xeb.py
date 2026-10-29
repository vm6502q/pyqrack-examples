# Coalesce nn_qb.py benchmark logs (files named like n10_d16_p2_b3.txt) into one CSV.
#
# By Dan Strano and (Anthropic) Claude.
#
# Usage: python3 coalesce_xeb.py [directory=.] [output=xeb_results.csv] [--summary summary.csv]
#
# Every file whose name contains n<width>_d<depth>_p<patches>_b<bonds> is read. Each run in a file is
# one block of "key: value" lines (circuit_seconds, ace_seconds, ideal_seconds, width, depth,
# max_bonds, patches, xeb_ace, hog_ace); device/OpenCL chatter is ignored. A block missing xeb_ace
# (e.g. a run cut off at the end of the file) is skipped and counted. One CSV row per run.
# With --summary, also writes one row per (n, d, p, b): run count, mean, sample std, standard error,
# and median of xeb_ace and hog_ace, plus mean ace_seconds.

import csv
import math
import re
import statistics
import sys
from pathlib import Path

NAME = re.compile(r"n(\d+)_d(\d+)_p(\d+)_b(\d+)")
LINE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)\s*$")
FIELDS = ["circuit_seconds", "ace_seconds", "ideal_seconds", "width", "depth", "max_bonds", "patches",
          "xeb_ace", "hog_ace"]
INTS = {"width", "depth", "max_bonds", "patches"}


def parse_runs(path):
    runs, cur, dropped = [], {}, 0

    def close():
        nonlocal dropped
        if cur:
            if "xeb_ace" in cur:
                runs.append(dict(cur))
            else:
                dropped += 1
            cur.clear()

    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.match(line)
            if not m or m.group(1) not in FIELDS:
                continue
            key, val = m.group(1), m.group(2)
            if key == "circuit_seconds" or key in cur:   # a new run starts
                close()
            cur[key] = int(float(val)) if key in INTS else float(val)
    close()
    return runs, dropped


def stats(xs):
    n = len(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else float("nan")
    return mean, sd, sd / math.sqrt(n) if n > 1 else float("nan"), statistics.median(xs)


def main(argv):
    args = list(argv)
    summary = None
    if "--summary" in args:
        i = args.index("--summary")
        summary = args[i + 1]
        del args[i:i + 2]
    root = Path(args[0]) if args else Path(".")
    out = args[1] if len(args) > 1 else "xeb_results.csv"

    files = []
    for p in root.iterdir():
        m = NAME.search(p.name)
        if p.is_file() and m:
            files.append((tuple(int(x) for x in m.groups()), p))
    files.sort()
    if not files:
        raise SystemExit(f"No files named like n<w>_d<d>_p<p>_b<b> in {root}")

    rows, groups = [], {}
    for (n, d, pp, b), path in files:
        runs, dropped = parse_runs(path)
        mism = 0
        for t, r in enumerate(runs):
            if any(r.get(k, v) != v for k, v in (("width", n), ("depth", d), ("patches", pp), ("max_bonds", b))):
                mism += 1
            rows.append({"file": path.name, "n": n, "d": d, "p": pp, "b": b, "trial": t,
                         **{k: r.get(k, "") for k in FIELDS}})
            groups.setdefault((n, d, pp, b), []).append(r)
        note = f", {dropped} incomplete run(s) skipped" if dropped else ""
        note += f", WARNING: {mism} run(s) disagree with the file name" if mism else ""
        print(f"{path.name}: {len(runs)} runs{note}", file=sys.stderr)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "n", "d", "p", "b", "trial"] + FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} runs from {len(files)} files to {out}", file=sys.stderr)

    if summary:
        cols = ["n", "d", "p", "b", "runs",
                "xeb_mean", "xeb_std", "xeb_sem", "xeb_median",
                "hog_mean", "hog_std", "hog_sem", "hog_median", "ace_seconds_mean"]
        with open(summary, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for key in sorted(groups):
                rs = groups[key]
                xs = stats([r["xeb_ace"] for r in rs])
                hs = stats([r["hog_ace"] for r in rs if "hog_ace" in r]) if any("hog_ace" in r for r in rs) else ("",) * 4
                at = [r["ace_seconds"] for r in rs if "ace_seconds" in r]
                w.writerow([*key, len(rs), *xs, *hs, statistics.fmean(at) if at else ""])
        print(f"Wrote {len(groups)} configurations to {summary}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
