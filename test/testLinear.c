#include "testLinear.h"


void TestForwardLinear(CuTest* tc)
{
    Linear* linear = createLinear(10, 20, true);
    int* shape = SHAPE(10);
    Tensor* input = createTensor(shape, 1, 1.0);
    Tensor* output = linear->baseLayer->forward((NNModule*)linear, input);
    PRINT(output);
}

void TestBackwardLinear(CuTest* tc)
{
    Linear* linear = createLinear(10, 20, true);
    Tensor* input = createTensor(SHAPE(10), 1, 1.0);
    Tensor* output = linear->baseLayer->forward((NNModule*)linear, input);

    Tensor* gradInput = createTensor(SHAPE(1, 20), 2, 1.0);
    Tensor* gradOutput = linear->baseLayer->backward((NNModule*)linear, gradInput);
    PRINT(gradOutput);
}

CuSuite* TestLinear(void)
{
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, TestForwardLinear);
    SUITE_ADD_TEST(suite, TestBackwardLinear);
    return suite;
}
