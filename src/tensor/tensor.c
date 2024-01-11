#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


Tensor* createTensor(int* shape, int dim, double val)
{
	Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
	if (tensor != NULL) {
        tensor->shape = (int*)malloc(dim * sizeof(int));
        if (tensor->shape != NULL) {
            memcpy(tensor->shape, shape, dim * sizeof(int));
        }

		tensor->dim = dim;

        // Compute stride and size
		tensor->stride = computeStride(shape, dim);
        tensor->size = computeSize(shape, dim);

        tensor->data = (double*)malloc(tensor->size * sizeof(double));

        if (tensor->data != NULL) {
            for (int i = 0; i < tensor->size; i++) {
                tensor->data[i] = val;
            }
        }

        // Assign methods
        tensor->getShape = getShape;
        tensor->getDim = getDim;
        tensor->getSize = getSize;

        tensor->_index = _index;
        tensor->setVal = setVal;
        tensor->getVal = getVal;

        tensor->reshape = reshape;
        tensor->permute = permute;
        tensor->print = printTensor;
	}
	return tensor;
}

void freeTensor(Tensor** ptrTensor)
{
    if (ptrTensor && *ptrTensor) {
        Tensor* tensor = *ptrTensor;
        free(tensor->shape);
        // Free the stride array
        free(tensor->stride);
        // Free the data array
        free(tensor->data);
        // Free the tensor structure itself
        free(tensor);
        *ptrTensor = NULL;
    }
}

int* computeStride(const int* shape, int size) {
    // Check for invalid input
    if (shape == NULL || size == 0) return NULL;
    int* stride = (int*)malloc(sizeof(int) * size);

    // Check for memory allocation failure
    if (stride == NULL) return NULL;
 
    for (int i = size - 1; i >= 0; i--){
        stride[i] = (i == size - 1) ? 1 :shape[i + 1] * stride[i + 1];
    }
    return stride;
}


int computeSize(const int* shape, int dim)
{
    if (shape == NULL) return 0;
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    return size;
}

int* getShape(const Tensor* self)
{
    return self->shape;
}

int getSize(const Tensor * self)
{
    return self->size;
}

int getDim(const Tensor* self)
{
    return self->dim;
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


void reshape(Tensor* self, const int* newShape, int newDim)
{
    int newSize = computeSize(newShape, newDim);
    assert(newSize == self->size);
    self->shape = (int*)malloc(newDim * sizeof(int));
    if (self->shape != NULL) {
        for (int i = 0; i < newDim; i++){
            self->shape[i] = newShape[i];
        }
    }
    self->dim = newDim;
    self->stride = computeStride(newShape, newDim);
    self->size = computeSize(newShape, newDim);

}

void permute(Tensor* self, const int* permuted, int dim)
{
    int *newShape = (int*)malloc(dim * sizeof(int));
    if (newShape != NULL) {
        for (int i = 0; i < dim; i++) {
            newShape[i] = self->shape[permuted[i]];
        }
    }
    self->stride = computeStride(newShape, dim);
    self->size = computeSize(newShape, dim);
}

void printTensor(Tensor* self)
{
    int *indices = (int*)malloc(self->dim *  sizeof(int));
    for (int i = 0; i < self->dim; i++) {
        indices[i] = 0;
    }

    prettyPrintTensor(self, indices, 0);
    printf("\n");
}

void prettyPrintTensor(const Tensor* self, int* indices, int currentDim) {
    if (currentDim == self->dim) {
        printf("%2.2lf ", self->getVal(self, indices));
    }
    else {
        printf("[ ");
        for (int i = 0; i < self->shape[currentDim]; i++) {
            indices[currentDim] = i;
            prettyPrintTensor(self, indices, currentDim + 1);
        }
        printf("] ");
    }
}