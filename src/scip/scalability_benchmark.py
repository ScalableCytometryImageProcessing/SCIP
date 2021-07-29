from scip.main import main as run_pipeline
import matplotlib.pyplot as plt
import pandas as pd
import time


def start_pipeline(n_workers, processes):
    path = os.environ["FULL_DATASET"]
    run_pipeline(
        paths=(path,),
        output_directory="tmp",
        headless=True,
        config='/home/sanderth/dask-pipeline/scip.yml',
        debug=True, n_workers=n_workers, processes=processes, port=8990, local=False)


if __name__ == "__main__":
    import os
    # add DEBUG_DATASET entry to terminal.integrated.env.linux in VS Code workspace settings
    # should contain path to small debug dataset
    iterations = 10
    time_df = pd.DataFrame(columns=["workers", "processes", "time"])
    n_workers = 1

    for processes in range(1, 3, 2):

        time_list = []

        for i in range(iterations):
            print(f"# proces: {processes}, iteration {i}")
            start = time.time()
            start_pipeline(n_workers=n_workers, processes=processes)
            end = time.time()
            time_list.append((end - start))

        average_time = sum(time_list) / len(time_list)
        time_df = time_df.append({"workers": n_workers, "processes": processes,
                                  "time": average_time}, ignore_index=True)

    time_df.to_csv('time_results.csv', index=False)
    plt.plot(time_df.processes, time_df.time)
    plt.savefig(f'benchmark_{n_workers}n_workers_process.pdf')
