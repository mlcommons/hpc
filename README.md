# MLPerf™ HPC reference implementations

This is a repository of reference implementations for the MLPerf HPC benchmarks.

General format should follow https://github.com/mlperf/training.

## Rules

The MLPerf HPC rules are based on the MLPerf Training rules with
some adjustments.

The MLPerf Training rules are available at [training\_rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc).

The MLPerf HPC specific rules are at [hpc\_training\_rules](https://github.com/mlcommons/training_policies/blob/master/hpc_training_rules.adoc).

## Compliance
The MLPerf logging package implements logging and compliance-checking utilities. This is available in hpc-1.0-branch of the MLPerf logging repository (https://github.com/mlcommons/logging/tree/hpc-1.0-branch).
These work for the HPC v2.0 submissions as well.

To install and test compliance of your runs/submissions:

```
# Install the package into your python environment.
# A development install (-e) is recommended for now so you can pull new updates.
git clone -b hpc-1.0-branch https://github.com/mlcommons/logging mlperf-logging
pip install [--user] -e mlperf-logging

# Test a full submission folder
python3 -m mlperf_logging.package_checker <YOUR SUBMISSION_FOLDER> hpc 1.0.0
```

There is also a script that performs compliance checks and summarizes the results. From the mlperf-logging directory (https://github.com/mlcommons/logging), use
```
./scripts/verify_for_v1.0_hpc.sh
```


