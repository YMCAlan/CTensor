#pragma once
#ifndef CONV_H
#define CONV_H

#include "../tensor/tensor.h"
#include "../nn/module.h"
typedef struct _conv Conv;

struct _conv
{
	// AbstractFactory for forward and backward
	NNModule* baseLayer;

	int in_channels;
	int out_channels;
	int* kernel_size;
	int* stride;
	int* padding;
	int* dilation;
	int* groups;
	char* padding_mode;

	Tensor* input;
	Tensor* weight;
	Tensor* bias;
	Tensor* gradInput;
	Tensor* gradWeight;
	Tensor* gradBias;

};
#endif // !CONV_H

