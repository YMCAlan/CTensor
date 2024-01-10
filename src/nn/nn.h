#pragma once
#ifndef NN_H
#define NN_H
#include "nn.h"
#include "../tensor/tensor.h"
#include <stdbool.h>
// Forward declaration
typedef struct nn NN;
typedef struct layer Layer;


struct nn
{
	int numsLayer;
	Layer* first;
	Layer* last;

	bool (*addLayer)(NN*, void*);
	Tensor* (*nnForward)(NN*, Tensor*);
};

struct layer
{
	void* layerPtr;
	Layer* next;
};

NN* createNN();
Layer* createLayer(void* m);

// add layer
bool addLayer(NN* nn, void* layerIn);

// forward method
Tensor* forward(NN*, Tensor*);

#endif // !NN_H
