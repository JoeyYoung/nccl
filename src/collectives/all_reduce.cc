/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "mlcc.h"


// todo, send to ccp ? or control in connectAddress, for local flow without setting ccp (not init problem)
void checkRankIndex(ncclComm* comm){
  // we only use one channel for each connection
  int myRank = comm->channels[0].ring.index;
  int nextRank = comm->channels[0].ring.next;
  int preRank = comm->channels[0].ring.prev;
  printf("[all_reduce.cc] One tensor will be sent from rank %d (ip:%s) to rank %d (ip: %s) with socket fd\n", myRank, myRankIP, nextRank, nextRankIP);
}

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

  // Judge whether this transmission cross machines
  // Make it Asyn, since only ncclSocketConnect not called at very first
  checkRankIndex(comm);

  return ncclEnqueueCheck(&info);
}