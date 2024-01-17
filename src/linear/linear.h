#pragma once
#ifndef LINEAR_H
#define LINEAR_H
#include "../tensor/tensor.h"
#include "../nn/module.h"
#include "../utils/macro.h"
#include <stdlib.h>

typedef struct _linear Linear;

struct _linear
{
	// AbstractFactory for forward and backward
	NNModule* nnM;

	int in_features;
	int out_features;

	Tensor* input;
	Tensor* weight;
	Tensor* bias;
	Tensor* gradInput;
	Tensor* gradWeight;
	Tensor* gradBias;

};

Linear* createLinear(int in_features, int out_features, bool bias);

Tensor* forwardLinear(NNModule* nnModule, Tensor* input);
Tensor* backwardLinear(NNModule* nnModule, Tensor* gradInput);
#endif // !LINEAR_H
