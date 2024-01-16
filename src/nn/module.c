#include "module.h"
#include <stdlib.h>

NNModule* createNNModule(ForwardFunction forward, BackwardFunction backward)
{
    NNModule* nnM = (NNModule*)malloc(sizeof(NNModule));
    if (nnM) {
        nnM->forward = forward;
        nnM->backward = backward;
    }
    return nnM;
}
