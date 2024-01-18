#include "nn.h"

NN* createNN()
{
	NN* nn = MALLOC(NN, 1);
	nn->modules = NULL;
	nn->numModules = 0;
	nn->addModule = addModule;
	nn->forward = forward;
	return nn;
}

void freeNN(NN** nnPtr)
{
	free(*nnPtr);
	*nnPtr = NULL;
}

void addModule(NN* nn, void* m)
{
	nn->modules = REALLOC(
		nn->modules,
		NNModule*,
		(nn->numModules + 1)
	);
	nn->modules[nn->numModules++] = m;

}

Tensor* forward(NN* nn, Tensor* input)
{
	Tensor* output = NULL;
	Tensor* tempInput = input->copy(input);
	for (int i = 0; i < nn->numModules; ++i) {
		Linear* tempModule = nn->modules[i];
		tempInput = tempModule->nnM->forward((NNModule*)tempModule, tempInput);
	}
	output = tempInput;
	return output;
}
