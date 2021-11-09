#include <stdio.h>
#include "socket.h"

/*
    Store ip, socket id info, help rank proc to judge and feed to ccp
*/

// ip of current rank and next rank, obtained from bootstrap
extern char* myRankIP; 
extern char* nextRankIP; // bug, the same as my rank

// the socket id used for tensor transmission, need to obtain for proxy->netsocket?
extern int* fdSend; // look net_socket / net.cc

// Rank has bug -> see trace_log, why show 4 ranks, all_Reduce number 有问题， 8 tensor -> 24op