#pragma once
#ifndef LINEAR_H
#define LINEAR_H
#include "../tensor/tensor.h"
typedef struct linear Linear;

struct linear
{
	int in_features;
	int out_features;
	Tensor* weight;
	Tensor* bias;
	Tensor* (*forward)(Linear*, Tensor*);
	void (*backward)(Linear*);
};

Linear* createLinear(int in_features, int out_features, bool bias);

Tensor* forwardLinear(Linear* self, Tensor* input);

#endif // !LINEAR_H
