import os
import re
import numpy as np
from numpy.typing import NDArray
import argparse

import plotly.graph_objects as go

from pathlib import Path


_benchmark_data = dict[str, dict[int, dict[str, NDArray[np.float64]]]] 
_indices_data = NDArray[np.float64]


def _read_benchmark_data(build_path: Path):
    data: _benchmark_data = {}
    indices: _indices_data | None = None

    for file_name in build_path.glob("src/benchmark_app/*.csv"):
        r = re.compile(r'^(\d+x\d+)_uint8_(.*).csv')

        matches = r.search(os.path.basename(file_name))
        if (matches is None):
            continue

        filter_size = matches.group(1)
        algo_name = matches.group(2)

        if filter_size not in data:
            data[filter_size] = {}

        a = np.loadtxt(file_name, delimiter='\t', skiprows=1)

        indices = np.unique(a[:, 0])
        processing_speed_split_by_mp = np.split(a[:,2], np.unique(a[:, 0], return_index=True)[1][1:])

        for num, mp in enumerate(np.unique(a[:, 0])):
            mp = int(mp)
            if mp not in data[filter_size]:
                data[filter_size][mp] = {}
            data[filter_size][mp][algo_name] = processing_speed_split_by_mp[num]

    return (data, indices)


def _generate_plots(data: _benchmark_data, indices: _indices_data, out_dir: Path):
    for mp in indices:
        fig = go.Figure()

        sorted_filter_sizes = sorted(list(data.keys()), key=lambda x:int(re.compile(r'^(\d+)').match(x).group(0)))
        algo_names = list(data[sorted_filter_sizes[0]][1].keys())
        data_by_algo = {k : {"y" : np.empty([0]), "x" : np.empty([0])} for k in algo_names}

        for algo_name in algo_names:
            for filter_size in sorted_filter_sizes:
                data_by_algo[algo_name]["y"] = np.append(data_by_algo[algo_name]["y"], data[filter_size][mp][algo_name])
                data_by_algo[algo_name]["x"] = np.append(data_by_algo[algo_name]["x"], np.full([np.size(data[filter_size][mp][algo_name])], filter_size))


        for algo_name in data_by_algo:
            fig.add_trace(go.Violin(
                y=data_by_algo[algo_name]["y"],
                x=data_by_algo[algo_name]["x"],
                name=algo_name,
                meanline_visible=False,
                bandwidth=20))

        fig.update_layout(
            yaxis_title='processing speed (MP/s)',
            xaxis_title='filter size (pixels)',
            template="plotly_dark",
            title=f'processing speeds of varying filter sizes for a {int(mp)} MP image'
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(out_dir / f"runtimes_{int(mp)}_mp.svg")


def _main(build_dir: Path, out_dir: Path) -> int:
    data, indices = _read_benchmark_data(build_dir)

    if len(data) == 0:
        raise ValueError(f"The path at {build_dir} does not seem to contain any benchmark data. Is the configuration correct and the benchmarks did run?")

    assert indices is not None
    _generate_plots(data, indices, out_dir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'cuda_median_filter plot generator')
    parser.add_argument('-build-dir', type=Path, default=(Path(__file__).parent.parent / "cmake-build-release").resolve(), help='path to the build dir (default: %(default)s)')
    parser.add_argument('-out-dir', type=Path, default=(Path(__file__).parent).resolve(), help='output dir (default: %(default)s)')
    args = parser.parse_args()
    _main(args.build_dir, args.out_dir)
