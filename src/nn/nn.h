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

typedef enum _layerType LayerType;
typedef union _genericLayer GenericLayer;

enum _layerType{
	LINEAR,
	CONV
};

union _genericLayer {
	Linear* linearLayer;
	Conv* convLayer;
};

struct _nnLayer {
	LayerType type;
	GenericLayer layer;
};


struct _nn
{
	int numsLayer;
	NNLayer* first;
	NNLayer* last;

	bool (*addLayer)(NN*, void*);
	Tensor* (*nnForward)(NN*, Tensor*);
};

NN* createNN();
void freeNN(void **nnPtr);
#endif // !NN_H
