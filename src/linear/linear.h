#pragma once
#ifndef LINEAR_H
#define LINEAR_H
#include "../tensor/tensor.h"
#include "../nn/module.h"
typedef struct linear Linear;

struct linear
{
	// AbstractFactory for forward and backward
	NNModule* baseLayer;

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
