#!/bin/bash

NCCL_ALGO=RING NCCL_PROTO=SIMPLE NCCL_NSOCKS_PERTHREAD=1 NCCL_MAX_NCHANNELS=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,NET,ENV NCCL_ALGO=RING horovodrun -np 2 -H 202.45.128.232:1,202.45.128.233:1 --start-timeout 300 --network-interface eno3 --verbose python pytorch_synthetic.py --model resnet50 > trace_log