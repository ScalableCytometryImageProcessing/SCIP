import pandas
import subprocess
from pathlib import Path
from datetime import datetime
import uuid
import json
from scip.utils import util
import logging


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    output = Path.cwd() / Path(f"benchmark_%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
    output.mkdir()

    paths = " ".join([
        "/home/maximl/shared_scratch/vulcan_pbmc_debug"
    ])
    iterations = 2    
    n_workers = 1

    timings = []
    for n_processes in range(1, 3, 2): 
        for i in range(iterations):

            timing = str(output / ("%s.json" % uuid.uuid4()))
            timings.append(timing)

            command = f"scip -j{n_workers} -n{n_processes} --no-local --headless --timing {timing} scip.yml {paths}"
            logger.info(command)
            subprocess.run(command, shell=True)

    timing_data =[]
    for timing in timings:
        with open(timing) as fp:
            timing_data.append(json.load(fp))
    pandas.DataFrame.from_records(timing_data).to_csv(str(output / 'timing_results.csv'), index=False)
