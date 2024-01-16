#pragma once

#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define INDEX(...) ((int[]){__VA_ARGS__})
#define SHAPE(...) ((int[]){__VA_ARGS__})
#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))

#define GET_VAL(t, ...) (t->getVal(t, ##__VA_ARGS__))
#define SET_VAL(t, val,...) (t->setVal(t, val, ##__VA_ARGS__))
#define PRINT(T) (T->print(T))
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
    int* (*getShape)(Tensor*); // Function to get the shape of the tensor
    int (*getDim)(Tensor*);    // Function to get the number of dimensions
    int (*getSize)(Tensor*);   // Function to get the total size of the tensor

    int (*_index)(Tensor*, int*);      // Function to calculate the index in the data array
    void (*setVal)(Tensor*, double, int*); // Function to set a value in the tensor
    double (*getVal)(Tensor*, int*);      // Function to get a value from the tensor

    Tensor* (*copy)(Tensor*);

    void (*reshape)(Tensor*, int*, int); // Function to reshape the tensor
    void (*permute)(Tensor*, int*, int); // Function to permute the tensor dimensions
    void (*transpose)(Tensor*);
    void (*print)(Tensor*); // Function to print the tensor
};

// Function to create or free a tensor
Tensor* createTensor(int* shape, int dim, double val);
void freeTensor(Tensor** ptrTensor);


int* computeStride(int * shape, int size);
int computeSize(int * shape, int dim);

// Access attribute methods
int* getShape(Tensor * self);
int getSize(Tensor * self);
int getDim(Tensor * self);

// Access element methods
int _index(Tensor* self, int * index);
void setVal(Tensor* self, double val, int * index);
double getVal(Tensor* self, int * index);

// Copy function
Tensor* copyTensor(Tensor* self);

// Shape manipulation methods
void reshape(Tensor* self, int * newShape, int newDim);
void permute(Tensor* self, int * permuted, int dim);
void transpose(Tensor* self);
void printTensor(Tensor* self);
void prettyPrintTensor(Tensor * self, int* indices, int currentDim);

// Operate function
// Matrix multiplication
Tensor* matMul(Tensor * input, Tensor * other);
Tensor* dotProduct(Tensor * input, Tensor * other);
Tensor* matProduct(Tensor * input, Tensor * other);

// Element-wise addition
Tensor* add(Tensor * input, Tensor * other);



#endif // !TENSOR_H