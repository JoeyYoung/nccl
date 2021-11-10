/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "mlcc.h"
#include "debug.h"

/*
fail just ignore socket id?
  (1) only cross machine will open shared memory
  (2) ccp always use the last socket flow as the reduce transmission.
*/

bool isCrossMachine(ncclComm* comm){
  // we only use one channel for each connection
  int myRank = comm->channels[0].ring.index;
  int nextRank = comm->channels[0].ring.next;
  int preRank = comm->channels[0].ring.prev;
  INFO(NCCL_NET, "[all_reduce.cc] Process data: from rank %d (ip:%s) to peer rank %d (ip: %s)", myRank, myRankIP, nextRank, nextRankIP);
  if (strcmp(myRankIP, nextRankIP) != 0) return true;
  else return false;
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

  // Judge whether this transmission cross machines, asyn?
  if(isCrossMachine(comm)){
    //todo, open shared memory to signal status
    printf("[all_reduce.cc] Open shared ....\n");
  }
  // do nothing if its intra node peer

  return ncclEnqueueCheck(&info);
}