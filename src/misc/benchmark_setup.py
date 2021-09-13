from pathlib import Path
from datetime import datetime
import uuid
import logging


def main():

    output = Path.cwd() / Path("benchmark_%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
    output.mkdir()
    (output / "results").mkdir()
    
    paths = " ".join([
        "/data/gent/vo/000/gvo00070/vsc42015/datasets/vib/vulban_pbmc"
    ])
    iterations = 3

    commands = []
    for partition_size in [10, 50, 100]:
        for n_threads, n_workers in zip(
            [2**i for i in range(5)], 
            reversed([2**i for i in range(5)])
        ):
            for _ in range(iterations): 
                ident = uuid.uuid4()
                timing = str(output / ("%s.json" % ident))

                o = str(output / "results" / str(ident))
                command = f'scip -j{n_workers} -t{n_threads} --no-local '
                command += f'-c28 -m64 -w00:45:00 -e "-A lt1_starter-245" '
                command += f'--headless --timing {timing} --partition-size {partition_size} '
                command += f'-d /local_scratch/ {o} scip.yml {paths}'

                commands.append(command)

    with open(str(output / "run.sh"), "w") as fh:
        fh.write("#!/bin/bash\n")
        for command in commands:
            fh.write(command+"\n")


if __name__ == "__main__":
    main()
