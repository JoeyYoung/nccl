/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "mlcc.h"

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };

  /*
    ncclMLCC, regard this transmission as the scheduling unit:
      judge whether the transmission needs to cross machines;
        open shared memory with ccpAgent:
            - singal to show which transmission is to be controlled;
        [ccpagent: send switch init scheudle, apply]
        [time slot: agent fetch info from swtich, reshcedule, apply]
      New tensor:
        continue ...
  */ 
  
  printf("one ncclAllReduce is trigged, ready to do ncclEnqueueCheck(&info).\n");
  hello_world();

  return ncclEnqueueCheck(&info);
}
