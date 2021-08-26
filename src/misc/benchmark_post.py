import pandas
from pathlib import Path
import json


def main(d):

    d = Path(d)

    timing_data = []
    for timing in d.glob("*.json"):
        with open(timing) as fp:
            timing_data.append(json.load(fp))
    pandas.DataFrame.from_records(timing_data).to_csv(
        str(d / 'timing_results.csv'), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str)

    args = parser.parse_args()

    main(args["dir"])