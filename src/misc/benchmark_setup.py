from pathlib import Path
from datetime import datetime
import uuid
import pandas 


def main():

    output = Path("/vsc-mounts/gent-user/420/vsc42015/vsc_data_vo/results/scip_benchmark")
    output = output / Path("benchmark_%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
    output.mkdir()
    (output / "results").mkdir()

    iterations = 5
    total_mem = 120

    commands = []
    for partition_size in [100, 200, 400, 800, 1600]:
        for n_workers in [1, 2, 4, 8, 16, 26]:
            for _ in range(iterations):
                ident = uuid.uuid4()

                o = str(output / "results" / str(ident))

                commands.append(dict(
                    n_workers=n_workers,
                    memory=total_mem // n_workers,
                    partition_size=partition_size,
                    output=o,
                    np=n_workers+2,
                    prefix=str(output)
                ))

    pandas.DataFrame(commands).to_csv(str(output / "data.csv"), index=False)

if __name__ == "__main__":
    main()
