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
    int* (*getShape)(const struct tensor*); // Function to get the shape of the tensor
    int (*getDim)(const struct tensor*);    // Function to get the number of dimensions
    int (*getSize)(const struct tensor*);   // Function to get the total size of the tensor

    int (*_index)(struct tensor*, const int*);      // Function to calculate the index in the data array
    void (*setVal)(struct tensor*, const int*, double); // Function to set a value in the tensor
    double (*getVal)(struct tensor*, const int*);      // Function to get a value from the tensor

    void (*print)(struct tensor*); // Function to print the tensor
};

Tensor* createTensor(int* shape, int dim, double val);

int* computeStride(int* shape, int size);
int computeSize(int* shape, int dim);

// access attribute method
int* getShape(const Tensor* self);
int getSize(const Tensor* self);
int getDim(const Tensor* self);


int _index(Tensor* self, const int * index);
void setVal(Tensor* self, const int * index, double val);
double getVal(Tensor* self, const int * index);
void printTensor(Tensor* self);
#endif // !TENSOR_H