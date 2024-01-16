#pragma once
#ifndef MODULE_H
#define MODULE
#include "../tensor/tensor.h"
typedef struct _nnModule NNModule;
typedef Tensor* (*ForwardFunction)(NNModule*, Tensor* input);
typedef Tensor* (*BackwardFunction)(NNModule*, Tensor* gradInput);

// Abstract base class for layers
struct _nnModule {
    ForwardFunction forward;
    BackwardFunction backward;
};

NNModule* createNNModule(
    ForwardFunction forward, 
    BackwardFunction backward
);

#endif // !MODULE_H
