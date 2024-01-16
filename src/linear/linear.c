#include "../linear/linear.h"
#include "../tensor/tensor.h"
#include <stdlib.h>


Linear* createLinear(int in_features, int out_features, bool bias)
{
	Linear* linear = (Linear*)malloc(sizeof(Linear));
	if (linear) {
		linear->input = NULL;
		linear->in_features = in_features;
		linear->out_features = out_features;
		
		linear->weight = createTensor(SHAPE(out_features, in_features), 2, 1.0);
		linear->gradWeight = createTensor(SHAPE(out_features, in_features), 2, 1.0);
		
		linear->bias = NULL;
		linear->gradBias = NULL;
		if (bias) {
			linear->bias = createTensor(SHAPE(out_features), 1, 1.0);
			linear->gradBias = createTensor(SHAPE(out_features), 1, 0.0);
		}
		linear->baseLayer = createNNModule(forwardLinear, backwardLinear);
	}
	return linear;
}

Tensor* forwardLinear(NNModule* nnModule, Tensor* input)
{
	// base-> derived type
	Linear* self = (Linear*)(nnModule);
	Tensor* weight = self->weight;
	Tensor* bias = self->bias;

	if (input->dim == 1) {
		int* newShape = SHAPE(1, input->shape[0]);
		input->reshape(input, newShape, COUNT_OF(newShape));
	}
	weight->transpose(weight);
	// input (1, n) * (n, m) -> (1, m)
	Tensor* output = matMul(input, weight);
	weight->transpose(weight);
	self->input = input->copy(input);
	return output;
}

/*
*/
Tensor* backwardLinear(NNModule* nnModule, Tensor* gradInput)
{
	// gradInput is (n * m)
	Linear* self = (Linear*)(nnModule);
	Tensor* weight = self->weight;
	Tensor* bias = self->bias;
	Tensor* input = self->input;

	Tensor* gradWeights = self->gradWeight; // size is (out, in)
	Tensor* gradBias = self->gradBias; // size is (out)

	// (1 * in) -> (in * 1)
	input->transpose(input);
	self->gradWeight = matMul(input, gradInput); // (in, 1) * (1, out);
	self->gradWeight->transpose(self->gradWeight);
	//clear local memory
	input = NULL;
	self->gradBias = createTensor(SHAPE(self->out_features), 1, 1.0);
	PRINT(self->gradBias);
	// (1, out) * (out, in)
	Tensor* gradOutput = matMul(gradInput, self->gradWeight);
	return gradOutput;
}
