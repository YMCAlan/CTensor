#include "testNN.h"

void TestCreateNN(CuTest* tc)
{
    NN* nn = createNN();

    CuAssertPtrNotNull(tc, nn);
    CuAssertIntEquals(tc, 0, nn->numModules);

    freeNN(&nn);
}

void TestAddLayer(CuTest* tc)
{
    NN* nn = createNN();
    CuAssertPtrNotNull(tc, nn);

    nn->addModule(nn, createLinear(10, 20, true));
    nn->addModule(nn, createLinear(20, 1, true));
    CuAssertIntEquals(tc, 2, nn->numModules);
    freeNN(&nn);
}

void TestForward(CuTest* tc)
{
    NN* nn = createNN();
    nn->addModule(nn, createLinear(10, 10, true));
    nn->addModule(nn, createLinear(10, 10, true));
    nn->addModule(nn, createLinear(10, 100, true));
    Tensor* input = createTensor(SHAPE(1, 10), 2, 1.0);
    Tensor* output = nn->forward(nn, input);
    PRINT_TENSOR(output);

}

CuSuite* TestNN(void)
{
    CuSuite* suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, TestCreateNN);
    SUITE_ADD_TEST(suite, TestAddLayer);
    SUITE_ADD_TEST(suite, TestForward);
    return suite;
}
