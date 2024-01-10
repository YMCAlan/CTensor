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
    // size and shape infomation
    int* shape;
    int* stride;
    int size; // total elements
    int dim;

    // Storage
    double* data;

    // Method 
    int (*getSize)(Tensor*);
    int (*getDim)(Tensor*);
    int* (*getShape)(Tensor*);

    int (*_index)(Tensor*, int*);
    double (*setVal)(Tensor*, int*, double);
    double (*getVal)(Tensor*, int*);

    void (*print)(Tensor*);
};

Tensor* createTensor(int* shape, int dim, double val);

int* computeStride(int* shape, int size);
int computeSize(int* shape, int dim);

// access attribute method
Tensor* getSelf(const Tensor* self);
int getSize(const Tensor* self);
int getDim(const Tensor* self);
int* getShape(const Tensor* self);

int _index(Tensor* self, const int * index);
void setVal(Tensor* self, const int * index, double val);
double getVal(Tensor* self, const int * index);
void printTensor(Tensor* self);
#endif // !TENSOR_H