#pragma once

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Forward declaration
typedef struct tensor Tensor;

struct tensor
{
    // Size and shape information
    int* shape;  // Array representing the dimensions of the tensor
    int dim;     // Number of dimensions
    int* stride; // Array representing the strides of the tensor
    int size;    // Total number of elements in the tensor

    // Storage
    double* data; // Array to store the actual data of the tensor

    // Methods
    int* (*getShape)(const Tensor*); // Function to get the shape of the tensor
    int (*getDim)(const Tensor*);    // Function to get the number of dimensions
    int (*getSize)(const Tensor*);   // Function to get the total size of the tensor

    int (*_index)(Tensor*, const int*);      // Function to calculate the index in the data array
    void (*setVal)(Tensor*, const int*, double); // Function to set a value in the tensor
    double (*getVal)(Tensor*, const int*);      // Function to get a value from the tensor

    void (*reshape)(Tensor*, const int*, int); // Function to reshape the tensor
    void (*permute)(Tensor*, const int*, int); // Function to permute the tensor dimensions
    void (*print)(struct tensor*); // Function to print the tensor
};

// Function to create or free a tensor
Tensor* createTensor(int* shape, int dim, double val);
void freeTensor(Tensor** ptrTensor);


int* computeStride(int* shape, int size);
int computeSize(int* shape, int dim);

// Access attribute methods
int* getShape(const Tensor* self);
int getSize(const Tensor* self);
int getDim(const Tensor* self);

// Access element methods
int _index(Tensor* self, const int * index);
void setVal(Tensor* self, const int * index, double val);
double getVal(Tensor* self, const int * index);

// Shape manipulation methods
void reshape(Tensor* self, const int* newShape, int newDim);
void permute(Tensor* self, const int* permuted, int dim);
void printTensor(Tensor* self);
void prettyPrintTensor(const Tensor* self, int* indices, int currentDim);

// Operate function
// Matrix multiplication
Tensor* matmul(const Tensor* input, const Tensor* other);
// Element-wise addition
Tensor* add(const Tensor* input, const Tensor* other);



#endif // !TENSOR_H