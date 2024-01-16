#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
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

        tensor->copy = copyTensor;
        tensor->reshape = reshape;
        tensor->permute = permute;
        tensor->transpose = transpose;
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

int* computeStride(int * shape, int size) {
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


int computeSize(int * shape, int dim)
{
    if (shape == NULL) return 0;
    int size = 1;
    for (int i = 0; i < dim; i++) {
        size *= shape[i];
    }
    return size;
}

int* getShape(Tensor * self)
{
    return self->shape;
}

int getSize(Tensor * self)
{
    return self->size;
}

int getDim(Tensor * self)
{
    return self->dim;
}


int _index(Tensor* self, int * index)
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


void setVal(Tensor* self, double val, int * index)
{
    int pos = self->_index(self, index);
    self->data[pos] = val;
}

double getVal(Tensor* self, int * index)
{
    int pos = self->_index(self, index);
    double val = 0.0;
    val = self->data[pos];
    return val;
}

Tensor* copyTensor(Tensor* original) {
    // Allocate memory for the new Tensor
    Tensor* newTensor = (Tensor*)malloc(sizeof(Tensor));
    if (newTensor == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Copy the non-dynamic members
    newTensor->dim = original->dim;
    newTensor->size = original->size;

    // Copy the shape
    newTensor->shape = (int*)malloc(sizeof(int) * original->dim);
    if (newTensor->shape == NULL) {
        // Handle memory allocation failure
        free(newTensor);
        return NULL;
    }
    memcpy(newTensor->shape, original->shape, sizeof(int) * original->dim);

    // Copy the stride
    newTensor->stride = (int*)malloc(sizeof(int) * original->dim);
    if (newTensor->stride == NULL) {
        // Handle memory allocation failure
        free(newTensor->shape);
        free(newTensor);
        return NULL;
    }
    memcpy(newTensor->stride, original->stride, sizeof(int) * original->dim);

    // Copy the data
    newTensor->data = (double*)malloc(sizeof(double) * original->size);
    if (newTensor->data == NULL) {
        // Handle memory allocation failure
        free(newTensor->shape);
        free(newTensor->stride);
        free(newTensor);
        return NULL;
    }
    memcpy(newTensor->data, original->data, sizeof(double) * original->size);

    // Copy the function pointers
    memcpy(&newTensor->getShape, &original->getShape, sizeof(original->getShape));
    memcpy(&newTensor->getDim, &original->getDim, sizeof(original->getDim));
    memcpy(&newTensor->getSize, &original->getSize, sizeof(original->getSize));
    memcpy(&newTensor->_index, &original->_index, sizeof(original->_index));
    memcpy(&newTensor->setVal, &original->setVal, sizeof(original->setVal));
    memcpy(&newTensor->getVal, &original->getVal, sizeof(original->getVal));
    memcpy(&newTensor->reshape, &original->reshape, sizeof(original->reshape));
    memcpy(&newTensor->permute, &original->permute, sizeof(original->permute));
    memcpy(&newTensor->transpose, &original->transpose, sizeof(original->transpose));
    memcpy(&newTensor->print, &original->print, sizeof(original->print));

    return newTensor;
}


void reshape(Tensor* self, int * newShape, int newDim)
{
    int newSize = computeSize(newShape, newDim);
    assert(newSize == self->size);
    self->shape = (int*)malloc(newDim * sizeof(int));
    if (self->shape != NULL) {
        for (int i = 0; i < newDim; i++){
            self->shape[i] = newShape[i];
        }
        self->dim = newDim;
        self->stride = computeStride(newShape, newDim);
        self->size = computeSize(newShape, newDim);
    }
    else {
        ;
    }
}

void permute(Tensor* self, int * permuted, int dim)
{
    int *newShape = (int*)malloc(dim * sizeof(int));
    if (newShape != NULL) {
        for (int i = 0; i < dim; i++) {
            newShape[i] = self->shape[permuted[i]];
        }
        // Free the previous stride array
        free(self->stride);
        // Update the shape, stride, and size
        self->shape = newShape;
        self->stride = computeStride(newShape, dim);
        self->size = computeSize(newShape, dim);
    }
    else
    {
        printf("Permute function alloc failed.");
    }
}

void transpose(Tensor* self)
{
    int* permuted = (int*)malloc(sizeof(int) * self->dim);
    for (int i = 0; i < self->dim; i++)
    {
        permuted[i] = self->dim - i - 1;
    }
    permute(self, permuted, self->dim);
}

void printTensor(Tensor* self)
{
    int *indices = (int*)malloc(self->dim *  sizeof(int));
    for (int i = 0; i < self->dim; i++) {
        indices[i] = 0;
    }
    printf("\n ========= Tensor ==========\n");
    prettyPrintTensor(self, indices, 0);
    printf("\n");
}

void prettyPrintTensor(Tensor * self, int* indices, int currentDim) {
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

Tensor* matMul(Tensor * input, Tensor * other)
{
    if (input->dim == 1 && other->dim == 1) {
        return dotProduct(input, other);
    }
    else if (input->dim == 2 && other->dim == 2)
    {
        return matProduct(input, other);
    }
    else
    {
        return NULL;
    }
}

Tensor* dotProduct(Tensor * input, Tensor * other)
{
    Tensor* result = createTensor(input->shape, input->dim, 0.0);
    for (int i = 0; i < input->shape[0]; i++)
    {
        SET_VAL(result, GET_VAL(input, i) + GET_VAL(other, i), i);
    }
    return result;
}

Tensor* matProduct(Tensor * input, Tensor * other)
{
    // Check dimensions for validity of matrix multiplication
    if (getShape(input)[1] != getShape(other)[0]) {
        // Dimensions mismatch, unable to perform matrix multiplication
        printf("Mat Product is Dimensions mismatch.\n");
        return NULL;
    }
    int m = getShape(input)[0]; // Number of rows in the result
    int p = getShape(input)[1]; // Number of columns in 'input' and rows in 'other'
    int n = getShape(other)[1]; // Number of columns in the result

    // Create a new tensor to store the result
    Tensor* result = createTensor(INDEX(m, n), 2, 0.0);

    // Perform matrix multiplication
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < p; ++k) {

                sum += input->getVal(input, INDEX(i, k)) * other->getVal(other, INDEX(k, j));
            }
            result->setVal(result, sum, INDEX(i, j));
        }
    }
    return result;
}

Tensor* add(Tensor * input, Tensor * other)
{
    // Determine the broadcast shape
    
    Tensor* result = NULL;
    return result;
}
