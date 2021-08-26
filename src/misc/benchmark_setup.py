from pathlib import Path
from datetime import datetime
import uuid
import logging


def main():

    output = Path.cwd() / Path("benchmark_%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
    output.mkdir()
    (output / "results").mkdir()
    
    logging.basicConfig(level=logging.INFO, filename=str(output / "benchmark.log"))
    logger = logging.getLogger(__name__)

    paths = " ".join([
        "/home/maximl/shared_scratch/vulcan_pbmc_debug"
    ])
    iterations = 3
    n_workers = 1

    commands = []
    for n_processes in [1, 2, 4, 8, 16, 32]:
        logger.info(f"Benchmarking {n_processes}")
        for i in range(iterations):
            logger.info(f"{n_processes}: iteration {i+1}/{iterations}")
            
            ident = uuid.uuid4()
            timing = str(output / ("%s.json" % ident))

            o = str(output / "results" / str(ident))
            command = f"scip -j{n_workers} -n{n_processes} --no-local "
            command += f"--headless --timing {timing} --partition-size 50"
            command += f"{o} scip.yml {paths}"

            commands.append(command)

    with open(str(output / "run.sh", "w")) as fh:
        fh.write("#!/bin/bash")
        for command in commands:
            fh.write(command)


if __name__ == "__main__":
    main()
