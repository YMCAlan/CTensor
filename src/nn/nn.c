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