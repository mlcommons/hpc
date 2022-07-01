# MLPerf™ HPC reference implementations

This is a repository of reference implementations for the MLPerf HPC benchmarks.

General format should follow https://github.com/mlperf/training.

## Rules

The MLPerf HPC rules are based on the MLPerf Training rules with
some adjustments.

The MLPerf Training rules are available at [training\_rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc).

The MLPerf HPC specific rules are at [hpc\_training\_rules](https://github.com/mlcommons/training_policies/blob/master/hpc_training_rules.adoc).

## Compliance
The MLPerf logging package implements logging and compliance-checking utilities
for MLPerf benchmarks in hpc-1.1.0 branch of the MLPerf logging repository (https://github.com/mlcommons/logging/tree/hpc-1.0-branch).
These work for the HPC v2.0 submissions as well.

To install and test compliance of your runs/submissions:

```
# Install the package into your python environment.
# A development install (-e) is recommended for now so you can pull new updates.
git clone -b hpc-1.0-branch https://github.com/mlperf-hpc/logging.git mlperf-logging
pip install [--user] -e mlperf-logging

# Test a full submission folder
python3 -m mlperf_logging.package_checker <YOUR SUBMISSION_FOLDER> hpc 1.0.0
```

There is also a script that tests both compliance and summarizes the results. From the mlperf-logging directory (https://github.com/mlcommons/logging), use
```
./scripts/verify_for_v1.0_hpc.sh
```

 <!-- 
## Compliance

 **IMPORTANT NOTE:** This section needs to be updated for HPC v1.0.
For now the referenced version of the `mlperf-logging` repository should be fine
for logging, but the compliance checking code and results summarizer code require
update. 

The MLPerf logging package implements logging and compliance-checking utilities
for MLPerf benchmarks. We have a temporary fork and hpc-0.5.0 branch in which we
are adding support for MLPerf-HPC v0.5 at
https://github.com/mlperf-hpc/logging/tree/hpc-0.5.0

To install and test compliance of your runs/submissions:
```
# Install the package into your python environment.
# A development install (-e) is recommended for now so you can pull new updates.
git clone -b hpc-0.5.0 https://github.com/mlperf-hpc/logging.git mlperf-logging
pip install [--user] -e mlperf-logging

# Test compliance of a specific mlperf hpc log file
python -m mlperf_logging.compliance_checker --ruleset hpc_0.5.0 $logFile

# Test a system description file (we just use the Training v0.7 rules)
python -m mlperf_logging.system_desc_checker $jsonFile training 0.7.0

# Test a full submission folder
python -m mlperf_logging.package_checker $submissionDir hpc 0.5.0
```
-->
