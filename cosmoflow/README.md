# CosmoFlow TensorFlow Keras benchmark implementation

This is a an implementation of the
[CosmoFlow](https://arxiv.org/abs/1808.04728) 3D convolutional neural network
for benchmarking. It is written in TensorFlow with the Keras API and uses
[Horovod](https://github.com/horovod/horovod) for distributed training.

You can find the previous TensorFlow implementation which accompanied the CosmoFlow paper at
https://github.com/NERSC/CosmoFlow

## Datasets

The dataset we use for this benchmark comes from simulations run by the
ExaLearn group and hosted at NERSC. The following web portal describes the
technical content of the dataset and provides links to the raw data.

https://portal.nersc.gov/project/m3363/

For this benchmark we currently use a preprocessed version of the dataset which
generates crops of size (128, 128, 128, 4) and stores in TFRecord format.
This preprocessing is done using the [prepare.py](prepare.py) script included
in this package. We describe here how to get access to this processed dataset,
but please refer to the ExaLearn web portal for additional technical details.

Globus is the current recommended way to transfer the dataset locally.
There is a globus endpoint at:

https://app.globus.org/file-manager?origin_id=d0b1b73a-efd3-11e9-993f-0a8c187e8c12&origin_path=%2F

The contents are also available via HTTPS at:

https://portal.nersc.gov/project/dasrepo/cosmoflow-benchmark/

The latest pre-processed dataset in TFRecord format is in the
`cosmoUniverse_2019_05_4parE_tf` folder, which contains training and validation
subfolders. There are currently 262144 samples for training and 65536 samples
for validation/testing. The combined size of the dataset is 5.1 TB.

For getting started, there is also a small tarball (179MB) with 32 training
samples and 32 validation samples, called `cosmoUniverse_2019_05_4parE_tf_small.tgz`.

For the previous dataset which was used for the 2020 ECP Annual Meeting results,
you can use the `cosmoUniverse_2019_02_4parE_dim128_cube_nT4.tar` tarball.
This is a 2.2 TB tar file containing 1027 `TFRecord` files, each representing
a simulated universe with 64 sub-volume samples.

## Running the benchmark

Submission scripts are in `scripts`. YAML configuration files go in `configs`.

### Running at NERSC

`sbatch -N 64 scripts/train_cori.sh`
