#!/bin/bash

NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_NSOCKS_PERTHREAD=1 NCCL_MAX_NCHANNELS=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET,ENV NCCL_ALGO=RING horovodrun -np 4 -H 202.45.128.232:2,202.45.128.233:2 --start-timeout 300 --network-interface eno3 --verbose python pytorch_mnist.py --epochs 1 --log-interval=1 --iterations=5 > trace_log
