import json

import numpy as np



# Explicit list of JSON filenames

files = [

    "Test_Results/single_domain/cvt/train0_test1_results_1.json",

    "Test_Results/single_domain/cvt/train0_test1_results_2.json",

    "Test_Results/single_domain/cvt/train0_test1_results_3.json",

    "Test_Results/single_domain/cvt/train0_test2_results_1.json",

    "Test_Results/single_domain/cvt/train0_test2_results_2.json",

    "Test_Results/single_domain/cvt/train0_test2_results_3.json",

    "Test_Results/single_domain/cvt/train0_test3_results_1.json",

    "Test_Results/single_domain/cvt/train0_test3_results_2.json",

    "Test_Results/single_domain/cvt/train0_test3_results_3.json",

]



# Metrics to compute

metrics = ["accuracy", "precision", "recall", "f1_score", "avg_inference_time_per_sample_sec"]

metric_values = {metric: [] for metric in metrics}



# Read each file and extract metric values

for file in files:

    with open(file, "r") as f:

        data = json.load(f)

        for metric in metrics:

            metric_values[metric].append(data[metric])



# Compute and display mean ± std

print("Aggregated Performance Metrics (mean ± std):\n")

for metric in metrics:

    values = np.array(metric_values[metric])

    mean = np.mean(values)

    std = np.std(values, ddof=1)  # sample standard deviation

    print(f"{metric:<15}: {mean:.4f} ± {std:.4f}")

