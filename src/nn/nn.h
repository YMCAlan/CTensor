#pragma once
#ifndef NN_H
#define NN_H

#include <stdbool.h>
#include <stdlib.h>

#include "module.h"
#include "../tensor/tensor.h"
#include "../linear/linear.h"
#include "../utils/macro.h"

// Forward declaration
typedef struct _nn NN;
typedef struct _layer Layer;

struct _nn
{
	int numsLayer;
	Layer* first;
	Layer* last;

	bool (*addLayer)(NN*, void*);
	Tensor* (*nnForward)(NN*, Tensor*);
};

struct _layer
{
	void* layerPtr;
	Layer* next;
};

NN* createNN();
void freeNN(void **nnPtr);
Layer* createLayer(void* m);

// add layer
bool addLayer(NN* nn, void* layerIn);

// forward method
Tensor* forward(NN*, Tensor*);
#endif // !NN_H
