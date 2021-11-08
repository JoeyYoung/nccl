#include <stdio.h>
#include "socket.h"

/*
    Store ip, socket id info, help rank proc to judge and feed to ccp
*/

// ip of current rank and next rank, obtained from bootstrap
extern char* myRankIP;
extern char* nextRankIP;

// the socket id used for tensor transmission, need to obtain for proxy->netsocket?
extern int* fdSend;