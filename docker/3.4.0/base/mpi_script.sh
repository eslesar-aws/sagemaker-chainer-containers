#!/usr/bin/env bash
touch /mpi_is_running &&
 PATH=/usr/local/nvidia/bin:/usr/local/openmpi/bin/:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/lib/python2.7/dist-packages/numpy/.libs:/usr/local/lib/python2.7/dist-packages/numpy/.libs:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/openmpi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH; python -m chainer_framework.run_training &&
 rm /mpi_is_running