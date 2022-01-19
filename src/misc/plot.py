import pandas
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq

if __name__ == "__main__":
    fs = Path('tmp/').glob("*.parquet")
    df = pandas.concat([pq.read_table(f).to_pandas() for f in fs])

    x = df["feat_area_combined"]
    y = df["feat_eccentricity_combined"]
    plt.scatter(x, y, s=1, alpha=0.5)
    plt.title(len(df))
    plt.savefig("test.png")
