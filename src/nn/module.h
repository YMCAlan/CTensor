#pragma once
#ifndef MODULE_H
#define MODULE

typedef void (*ForwardFunction)(void*);
typedef void (*BackwardFunction)(void*);
typedef struct nnModule NNModule;

// Abstract base class for layers
struct nnModule {
    ForwardFunction forward;
    BackwardFunction backward;
};

NNModule* createNNModule(ForwardFunction forward, BackwardFunction backward);

#endif // !MODULE_H
