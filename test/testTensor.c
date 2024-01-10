#include "testTensor.h"
#include "cuTest.h"
#include "../src/tensor/tensor.h"

void testComputeStride()
{
    int shape[] = { 2, 3, 4 };  // Example shape
    int size = 3;             // Example size

    // Call the computeStride function
    int* stride = computeStride(shape, size);

    // Check if the result is not NULL
    CU_ASSERT_PTR_NOT_NULL(stride);
    // Check if the stride values are correct
    CU_ASSERT_EQUAL(stride[0], 12);
    CU_ASSERT_EQUAL(stride[1], 4);
    CU_ASSERT_EQUAL(stride[2], 1);

    // Clean up
    free(stride);
}

CuSuite* TestTensor(void)
{
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, testComputeStride);
	return suite;
}
