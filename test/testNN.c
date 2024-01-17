#include "testNN.h"

void TestCreateNN(CuTest* tc)
{
    NN* nn = createNN();

    CuAssertPtrNotNull(tc, nn);
    CuAssertIntEquals(tc, 0, nn->numsLayer);
    CuAssertPtrEquals(tc, NULL, nn->first);
    CuAssertPtrEquals(tc, NULL, nn->last);

    freeNN(&nn);
}

void TestAddLayer(CuTest* tc)
{
    NN* nn = createNN();
    CuAssertPtrNotNull(tc, nn);

    nn->addLayer(nn, createLinear(10, 20, true));
    nn->addLayer(nn, createLinear(20, 30, true));
    CuAssertIntEquals(tc, 2, nn->numsLayer);
    CuAssertPtrNotNull(tc, nn->first);
    freeNN(&nn);
}

void TestForward(CuTest* tc)
{
    NN* nn = createNN();
    nn->addLayer(nn, createLinear(10, 20, true));
    nn->addLayer(nn, createLinear(20, 30, true));

    Tensor* input = createTensor(SHAPE(1, 10), 1, 1.0);
    Tensor* output = nn->nnForward(nn, input);
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
