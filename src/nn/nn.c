#include "nn.h"

NN* createNN()
{
	NN* nn = MALLOC(NN, 1);
	nn->numsLayer = 0;
	nn->first = NULL;
	nn->last = NULL;
	nn->addLayer = addLayer;
	nn->nnForward = forward;
	return nn;
}

void freeNN(void** nnPtr)
{
	free(*nnPtr);
}

Layer* createLayer(void* m)
{
	Layer* layer = MALLOC(Layer, 1);
	layer->layerPtr = m;
	layer->next = NULL;
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
		output= ((NNModule*)(currLayer->layerPtr))->forward(currLayer->layerPtr, input);
		input = output;
		currLayer = currLayer->next;
	}
	return output;
}