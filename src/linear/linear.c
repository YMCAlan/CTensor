#include "../linear/linear.h"
#include "../tensor/tensor.h"
#include <stdlib.h>


Linear* createLinear(int in_features, int out_features, bool bias)
{
	Linear* linear = (Linear*)malloc(sizeof(Linear));
	if (linear) {
		linear->in_features = in_features;
		linear->out_features = out_features;
		
		int weightShape[] = { out_features, in_features };
		linear->weight = createTensor(weightShape, 2, 0);
		linear->gradWeight = createTensor(weightShape, 2, 0);

		int biasShape[] = { out_features };
		linear->bias = NULL;
		linear->gradBias = NULL;
		if (bias) {
			linear->bias = createTensor(biasShape, 1, 1.0);
			linear->gradBias = createTensor(biasShape, 1, 0.0);
		}
	}
	return linear;
}

Tensor* forwardLinear(Linear* self, Tensor* input)
{
	Tensor* output = createTensor((int[]) { self->out_features }, 1, 0.0);
	Tensor* weight = self->weight;
	Tensor* bias = self->bias;
	double x = 0.0, w = 0.0;

	for (int i = 0; i < self->out_features; i++) {
		double sumValue = 0.0;
		for (int j = 0; j < self->in_features; j++) {
			// y = x * w
			x = input->getVal(input, (int[]){ j });
			w = weight->getVal(weight, (int[]) {i, j});
			sumValue = sumValue + (x * w);
		}
		if (bias != NULL) {
			sumValue += bias->getVal(bias, (int[]) { i });
		}
		output->setVal(output, (int[]) { i }, sumValue);
	}
	return output;
}
