#pragma once
#ifndef LINEAR_H
#define LINEAR_H
#include "../tensor/tensor.h"
#include "../nn/module.h"
typedef struct linear Linear;

struct linear
{
	// AbstractFactory for forward and backward
	NNModule base;

	int in_features;
	int out_features;

	Tensor* weight;
	Tensor* bias;
	Tensor* gradInput;
	Tensor* gradWeight;
	Tensor* gradBias;

};

Linear* createLinear(int in_features, int out_features, bool bias);

Tensor* forwardLinear(Linear* self, Tensor* input);

#endif // !LINEAR_H
