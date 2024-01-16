#include "testTensor.h"
#include "cuTest.h"
#include "../src/tensor/tensor.h"
#include "../src/linear/linear.h"

void TestCreateTensor(CuTest* tc) {
    // Test Case 1: Create a tensor with a shape of {2, 3}, dimension 2, and initial value 1.0
    int shape1[] = { 2, 3 };
    int dim1 = 2;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);
    CuAssertPtrNotNull(tc, tensor1->shape);
    CuAssertPtrNotNull(tc, tensor1->data);
    CuAssertIntEquals(tc, dim1, tensor1->dim);
    CuAssertIntEquals(tc, shape1[0], tensor1->shape[0]);
    CuAssertIntEquals(tc, shape1[1], tensor1->shape[1]);
    CuAssertIntEquals(tc, shape1[0] * shape1[1], tensor1->size);

    for (int i = 0; i < tensor1->size; i++) {
        CuAssertDblEquals(tc, val1, tensor1->data[i], 0.001);
    }

    // Test Case 2: Create a tensor with a shape of {3, 4, 5}, dimension 3, and initial value 2.5
    int shape2[] = { 3, 4, 5 };
    int dim2 = 3;
    double val2 = 2.5;
    Tensor* tensor2 = createTensor(shape2, dim2, val2);

    CuAssertPtrNotNull(tc, tensor2);
    CuAssertPtrNotNull(tc, tensor2->shape);
    CuAssertPtrNotNull(tc, tensor2->data);
    CuAssertIntEquals(tc, dim2, tensor2->dim);
    CuAssertIntEquals(tc, shape2[0], tensor2->shape[0]);
    CuAssertIntEquals(tc, shape2[1], tensor2->shape[1]);
    CuAssertIntEquals(tc, shape2[2], tensor2->shape[2]);
    CuAssertIntEquals(tc, shape2[0] * shape2[1] * shape2[2], tensor2->size);

    for (int i = 0; i < tensor2->size; i++) {
        CuAssertDblEquals(tc, val2, tensor2->data[i], 0.001);
    }

    // Cleanup
    freeTensor(&tensor1);
    freeTensor(&tensor2);
}

// Function to test computeStride
void TestComputeStride(CuTest* tc) {
    // Test Case 1: Valid input with size 3
    int shape1[] = { 2, 3, 4 };
    int size1 = 3;
    int* stride1 = computeStride(shape1, size1);

    CuAssertPtrNotNull(tc, stride1);
    CuAssertIntEquals(tc, 12, stride1[0]);
    CuAssertIntEquals(tc, 4, stride1[1]);
    CuAssertIntEquals(tc, 1, stride1[2]);

    // Test Case 2: Valid input with size 2
    int shape2[] = { 5, 6 };
    int size2 = 2;
    int* stride2 = computeStride(shape2, size2);

    CuAssertPtrNotNull(tc, stride2);
    CuAssertIntEquals(tc, 6, stride2[0]);
    CuAssertIntEquals(tc, 1, stride2[1]);

    // Test Case 3: Invalid input (NULL shape)
    int size3 = 3;
    int* stride3 = computeStride(NULL, size3);

    CuAssertPtrEquals(tc, NULL, stride3);

    // Clean up
    free(stride1);
    free(stride2);
}

// Function to test computeSize
void TestComputeSize(CuTest* tc) {
    // Test Case 1: Valid input with dimension 3
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    int size1 = computeSize(shape1, dim1);

    CuAssertIntEquals(tc, 24, size1);
     
    // Test Case 2: Valid input with dimension 2
    int shape2[] = { 5, 6 };
    int dim2 = 2;
    int size2 = computeSize(shape2, dim2);

    CuAssertIntEquals(tc, 30, size2);

    // Test Case 3: Invalid input (NULL shape)
    int dim3 = 3;
    int size3 = computeSize(NULL, dim3);

    CuAssertIntEquals(tc, 0, size3);
}

void TestGetShape(CuTest* tc) {
    // Test Case 1: Valid input
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);
    CuAssertPtrNotNull(tc, getShape(tensor1));

    // Verify shape values
    for (int i = 0; i < dim1; i++) {
        CuAssertIntEquals(tc, shape1[i], getShape(tensor1)[i]);
    }

    // Clean up
    freeTensor(&tensor1);
}

// Function to test getSize
void TestGetSize(CuTest* tc) {
    // Test Case 1: Valid input
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);
    CuAssertIntEquals(tc, dim1, getDim(tensor1));
    CuAssertIntEquals(tc, computeSize(shape1, dim1), getSize(tensor1));

    // Clean up
    freeTensor(&tensor1);
}

// Function to test getDim
void TestGetDim(CuTest* tc) {
    // Test Case 1: Valid input
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);
    CuAssertIntEquals(tc, dim1, getDim(tensor1));

    // Clean up
    freeTensor(&tensor1);
}

// Function to test _index, setVal, getVal
void TestIndexSetGet(CuTest* tc) {
    // Test Case 1: Valid input
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);

    int index[] = { 1, 2, 3 };
    int expectedPosition = 1 * tensor1->stride[0] + 2 * tensor1->stride[1] + 3 * tensor1->stride[2];

    CuAssertIntEquals(tc, expectedPosition, _index(tensor1, index));

    setVal(tensor1, 5.0, index);
    CuAssertDblEquals(tc, 5.0, getVal(tensor1, index), 0.001);

    // Clean up
    freeTensor(&tensor1);
}

// Function to test reshape
void TestReshape(CuTest* tc) {
    // Test Case 1: Valid input
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);

    int newShape[] = { 3, 4, 2 };
    int newDim = 3;
    reshape(tensor1, newShape, newDim);

    CuAssertIntEquals(tc, newDim, getDim(tensor1));
    CuAssertIntEquals(tc, computeSize(newShape, newDim), getSize(tensor1));

    // Clean up
    freeTensor(&tensor1);
}

// Function to test permute
void TestPermute(CuTest* tc) {
    // Test Case 1: Valid input
    int shape1[] = { 2, 3, 4 };
    int dim1 = 3;
    double val1 = 1.0;
    Tensor* tensor1 = createTensor(shape1, dim1, val1);

    CuAssertPtrNotNull(tc, tensor1);

    int permuted[] = { 2, 0, 1 };
    int expectedShape[] = { 4, 2, 3 };
    int newDim = 3;
    permute(tensor1, permuted, newDim);

    CuAssertIntEquals(tc, newDim, getDim(tensor1));
    CuAssertIntEquals(tc, computeSize(expectedShape, newDim), getSize(tensor1));

    // Clean up
    freeTensor(&tensor1);
}


CuSuite* TestTensor(void)
{
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, TestCreateTensor);
    SUITE_ADD_TEST(suite, TestComputeStride);
    SUITE_ADD_TEST(suite, TestComputeSize);
    SUITE_ADD_TEST(suite, TestGetShape);
    SUITE_ADD_TEST(suite, TestGetSize);
    SUITE_ADD_TEST(suite, TestGetDim);
    SUITE_ADD_TEST(suite, TestIndexSetGet);
    SUITE_ADD_TEST(suite, TestReshape);
    SUITE_ADD_TEST(suite, TestPermute);
	return suite;
}
