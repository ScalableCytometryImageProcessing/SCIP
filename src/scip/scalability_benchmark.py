import pandas
from pathlib import Path
from datetime import datetime
import uuid
import json
import logging
import subprocess
import shlex


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

    timings = []
    for n_processes in [1, 2, 4, 8, 16]:
        logger.info(f"Benchmarking {n_processes}")
        for i in range(iterations):
            logger.info(f"{n_processes}: iteration {i+1}/{iterations}")
            
            ident = uuid.uuid4()
            timing = str(output / ("%s.json" % ident))

            o = str(output / "results" / str(ident))
            command = f"scip -j{n_workers} -n{n_processes} --no-local "
            command += f"--headless --timing {timing} -o {o} scip.yml {paths}"

            logger.info(f"Launching: {command}")
            ret = subprocess.run(shlex.split(command))
            timings.append((ret.returncode, timing))

    timing_data = []
    for ret, timing in timings:
        if ret == 0:
            with open(timing) as fp:
                timing_data.append(json.load(fp))
    pandas.DataFrame.from_records(timing_data).to_csv(
        str(output / 'timing_results.csv'), index=False)


if __name__ == "__main__":
    main()
