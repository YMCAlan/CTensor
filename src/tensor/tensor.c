#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

Tensor* createTensor(int* shape, int dim, double val)
{
	Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
	if (tensor) {
        tensor->shape = (int*)malloc(dim * sizeof(int));
        if (tensor->shape != NULL) {
            for (int i = 0; i < dim; i++) {
                tensor->shape[i] = shape[i];
            }
        }

		tensor->dim = dim;
		tensor->stride = computeStride(shape, dim);
        tensor->size = computeSize(shape, dim);

        tensor->data = (double*)malloc(tensor->size * sizeof(double));
        
        if (tensor->data != NULL) {
            for (int i = 0; i < tensor->size; i++) {
                tensor->data[i] = val;
            }
        }

        // method
        tensor->getSize = getSize;
        tensor->getDim = getDim;
        tensor->getShape = getShape;

        tensor->_index = _index;
        tensor->setVal = setVal;
        tensor->getVal = getVal;
        tensor->print = printTensor;
	}
	return tensor;
}

int* computeStride(const int* shape, int size) {
    // Check for invalid input
    if (shape == NULL || size == 0) return NULL;

    int* stride = (int*)malloc(sizeof(int) * size);

    // Check for memory allocation failure
    if (stride == NULL) return NULL;
 
    for (int i = size - 1; i >= 0; i--)
    {
        stride[i] = (i == size - 1) ? 1 :shape[i + 1] * stride[i + 1];
    }
    return stride;
}


int computeSize(int* shape, int dim)
{
    if (shape == NULL) return 0;

    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }

    return size;
}


Tensor* getSelf(const Tensor* self)
{
    return self;
}

int getSize(const Tensor * self)
{
    return self->size;
}

int getDim(const Tensor* self)
{
    return self->dim;
}

int* getShape(const Tensor* self)
{
    return self->shape;
}

int _index(Tensor* self, const int * index)
{   
    for (int i = 0; i < self->dim; i++) {
        if (index[i] < 0 || index[i] >= self->shape[i]) {
            printf("Index out of range!\n");
            return -1;
        }
    }

    int position = 0;
    for (int k = 0; k < self->dim; k++) {
        position += (self->stride[k] * index[k]);
    }
    return position;
}


void setVal(Tensor* self, const int * index, double val)
{
    int pos = self->_index(self, index);
    self->data[pos] = val;
}

double getVal(Tensor* self, const int * index)
{
    int pos = self->_index(self, index);
    double val = 0.0;
    val = self->data[pos];
    return val;
}

void printTensor(Tensor* self)
{
    for (int i = 0; i < self->size; i++) {
        printf("Tensor[%d] = %2.5lf\n", i, self->data[i]);
    }
    printf("\n\n");
}
