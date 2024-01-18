#pragma once
#ifndef NN_H
#define NN_H

#include <stdbool.h>
#include <stdlib.h>

#include "module.h"
#include "../tensor/tensor.h"
#include "../linear/linear.h"
#include "../conv/conv.h"
#include "../utils/macro.h"

// Forward declaration
typedef struct _nn NN;
typedef struct _nnLayer NNLayer;

struct _nn
{
	int numModules;
	NNModule** modules;
	
	void (*addModule)(NN*, void*);
	Tensor* (*forward)(NN*, Tensor*);
};

NN* createNN();
void freeNN(NN** nnPtr);
void addModule(NN* nn, void* m);

Tensor* forward(NN* nn, Tensor* input);
#endif // !NN_H
