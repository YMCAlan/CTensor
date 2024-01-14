#pragma once
#ifndef MODULE_H
#define MODULE


typedef struct nnModule NNModule;

// Abstract base class for layers
struct nnModule {
    void (*forward)(void*, void*);
    void (*backward)(void*, void*);
};



#endif // !MODULE_H
