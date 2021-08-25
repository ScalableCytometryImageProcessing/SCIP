import pandas
from pathlib import Path
from datetime import datetime
import uuid
import json
import logging
import asyncio


async def main():

    output = Path.cwd() / Path("benchmark_%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
    o = str(output / "tmp")
    output.mkdir()
    
    logging.basicConfig(level=logging.INFO, filename=str(output / "benchmark.log"))
    logger = logging.getLogger(__name__)

    paths = " ".join([
        "/home/maximl/shared_scratch/vulcan_pbmc"
    ])
    iterations = 5
    n_workers = 1

    timings = []
    subprocs = []
    for n_processes in [1, 2, 4, 8, 16, 32]:
        logger.info(f"Benchmarking {n_processes}")
        for i in range(iterations):
            logger.info(f"{n_processes}: iteration {i+1}/{iterations}")

            timing = str(output / ("%s.json" % uuid.uuid4()))
            timings.append(timing)

            command = f"scip -j{n_workers} -n{n_processes} --no-local "
            command += f"--headless --timing {timing} -o {o} scip.yml {paths}"

            logger.info(f"Launching: {command}")
            proc = await asyncio.create_subprocess_shell(command)
            subprocs.append(proc.wait())

            if len(subprocs) >= 6:
                logger.info(f"Waiting for next {len(subprocs)} tasks to finish")
                await asyncio.gather(*subprocs)
                subprocs = []

    logger.info(f"Waiting for final {len(subprocs)} tasks to finish")
    await asyncio.gather(*subprocs)

    timing_data = []
    for timing in timings:
        with open(timing) as fp:
            timing_data.append(json.load(fp))
    pandas.DataFrame.from_records(timing_data).to_csv(
        str(output / 'timing_results.csv'), index=False)


if __name__ == "__main__":
    asyncio.run(main())
