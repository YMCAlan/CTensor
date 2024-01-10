#include "nn.h"
#include "../tensor/tensor.h"
#include "../linear/linear.h"
#include <stdlib.h>
NN* createNN()
{
	NN* nn = (NN*)malloc(sizeof(NN));
	if (nn) {
		nn->numsLayer = 0;
		nn->first = NULL;
		nn->last = NULL;
		nn->addLayer = addLayer;
		nn->nnForward = forward;
	}
	return nn;
}

Layer* createLayer(void* m)
{
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	if (layer) {
		layer->layerPtr = m;
		layer->next = NULL;
	}
	return layer;
}

bool addLayer(NN* nn, void* layerIn)
{
	Layer* layer = createLayer(layerIn);
	if (!nn || !layer) {
		return false;
	}

	if (nn->numsLayer == 0) {
		nn->first = layer;
	}
	else {
		nn->last->next = layer;
	}
	nn->last = layer;
	(nn->numsLayer)++;
	return true;
}

Tensor* forward(NN* nn, Tensor* input)
{
	Layer* currLayer = nn->first;
	Tensor* output = NULL;
	while (currLayer != NULL) {
		Linear* linear = (Linear*)(currLayer->layerPtr);
		output = linear->forward(linear, input);
		input = output;
		currLayer = currLayer->next;
	}
	return output;
}