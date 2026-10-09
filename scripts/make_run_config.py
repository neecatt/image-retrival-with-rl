#!/usr/bin/env python3
"""Write a run-specific config: base YAML plus dotted-key overrides. The base file is not modified.

Example:
  make_run_config.py config/config.yaml "$TMPDIR/run.yaml" \
      dataset.root_dir="$TMPDIR/data/raw" training.stage2.max_batches_per_epoch=50
Values are parsed as YAML scalars (50 -> int, true -> bool, quoted text -> str).
"""
import sys
import yaml


def main(base, out, *overrides):
    with open(base) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, raw = item.partition('=')
        if not key or not _:
            sys.exit(f"bad override {item!r}, expected key.path=value")
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            if p not in node or not isinstance(node[p], dict):
                sys.exit(f"unknown config section {p!r} in {key!r}")
            node = node[p]
        node[leaf] = yaml.safe_load(raw)
        print(f"  {key} = {node[leaf]!r}")
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {out}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
