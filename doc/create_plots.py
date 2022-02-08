import os
import glob
import re
import numpy as np

import plotly.graph_objects as go

data = {}
indices = []

for file_name in glob.glob("../cmake-build-release/src/benchmark_app/*.csv"):
    r = re.compile(r'^(\d+x\d+)_uint8_(.*).csv')

    matches = r.search(os.path.basename(file_name))
    if (matches is None):
        continue

    filter_size = matches.group(1)
    algo_name = matches.group(2)

    if filter_size not in data:
        data[filter_size] = {}

    a = np.loadtxt(file_name, delimiter='\t', skiprows=1)

    indices =  np.unique(a[:, 0])
    processing_speed_split_by_mp = np.split(a[:,2], np.unique(a[:, 0], return_index=True)[1][1:])

    for num, mp in enumerate(np.unique(a[:, 0])):
        if mp not in data[filter_size]:
            data[filter_size][mp] = {}
        data[filter_size][mp][algo_name] = processing_speed_split_by_mp[num]

for mp in indices:

    fig = go.Figure()

    data_by_algo = {}

    for algo_name in data[filter_size][mp]:
        if algo_name not in data_by_algo:
            data_by_algo[algo_name] = {"y" : np.empty([0]), "x" : np.empty([0])}

        sorted_filter_sizes = sorted(list(data.keys()), key=lambda x:int(re.compile(r'^(\d+)').match(x).group(0)))

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

    fig.write_image(f"runtimes_{int(mp)}_mp.svg")
