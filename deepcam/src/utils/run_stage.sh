#!/bin/bash

# data dir
mode=$1
src_dir=$2
dest_dir=$3

# step into source dir
cd $(dirname ${src_dir})

# tar
if [ "${mode}" == "tar" ]; then
    /opt/mpifileutils/bin/dtar -cf $(basename ${src_dir}).tar $(basename ${src_dir})
fi

# zip
if [ "${mode}" == "compress" ]; then
    /opt/mpifileutils/bin/dbz2 -z $(basename ${src_dir}).tar
fi

# bcast
if [ "${mode}" == "broadcast" ]; then
    if [ -f $(basename ${src_dir}).dbz2 ]; then
	srcfile=$(basename ${src_dir}).tar.dbz2
    else
	srcfile=$(basename ${src_dir}).tar
    fi
    /opt/mpifileutils/bin/dbcast ${srcfile} ${dest_dir}/$(basename ${srcfile})
fi

# untar
if [ "${mode}" == "untar" ]; then
    local_rank=$(( ${PMIX_RANK} % 8 ))
    if [ "${local_rank}" == "0" ]; then
	time tar -xf ${dest_dir}/$(basename ${src_dir}).tar
    fi
fi
