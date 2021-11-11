/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "mlcc.h"
#include "debug.h"
#include "core.h"

struct modelSize accumlSize;

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
  INFO(NCCL_NET, "[all_reduce.cc] Process data: from rank %d (ip:%s) to peer rank %d (ip: %s).", myRank, myRankIP, nextRank, nextRankIP);
  if (strcmp(myRankIP, nextRankIP) != 0) return true;
  else return false;
}

void accumulateTensorSize(ncclDataType_t datatype, size_t count){
  size_t dataSize = (size_t) ncclTypeSize(datatype) * count; // represented as bytes
  accumlSize.B += dataSize;
  // tune accumlSize to fit
  if(accumlSize.B >= 1024){
    accumlSize.M += (accumlSize.B/1024);
    accumlSize.B %= 1024;
  }
  if(accumlSize.M >= 1024){
    accumlSize.G += (accumlSize.M/1024);
    accumlSize.M %= 1024;
  }
  INFO(NCCL_NET, "[all_reduce.cc] accumulate data sent: %zuG %zuM %zuB.", accumlSize.G, accumlSize.M, accumlSize.B);
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };

  // ncclMLCC, Judge whether this transmission cross machines
  if(isCrossMachine(comm)){
    // accumulate total data size transmitted
    accumulateTensorSize(datatype, count);
    //todo, open shared memory, store
    printf("[all_reduce.cc] Open shared ....\n");
  }
  // do nothing if its intra node peer

  // DL manages allreduce op as fifo, enqueue to kernel when call nccl op.
  // However, nccl use its own sized fifo to perform actual communication, which is hard to track
  // we argue: tensor level = iter level, just see transmission between two ranks in one iter as one ccp flow. 
  return ncclEnqueueCheck(&info);
}