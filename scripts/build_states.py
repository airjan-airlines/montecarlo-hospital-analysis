"""raw eICU demo tables -> data/processed/{trajectories,transitions}.parquet"""

import argparse
import logging

from mtraj.config import load_config
from mtraj.load import load_raw
from mtraj.states import build_trajectories, transitions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    traj = build_trajectories(load_raw(cfg["data"]["root"]), cfg)
    out = cfg["data"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    traj.to_parquet(out / "trajectories.parquet", index=False)
    transitions(traj).to_parquet(out / "transitions.parquet", index=False)
    logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
