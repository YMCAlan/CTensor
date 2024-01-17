#pragma once
#ifndef TEST_NN
#define TEST_NN

#include "cuTest.h"
#include "../src/tensor/tensor.h"
#include "../src/linear/linear.h"
#include "../src/nn/nn.h"

void TestCreateNN(CuTest* tc);
void TestAddLayer(CuTest* tc);
void TestForward(CuTest* tc);
CuSuite* TestNN(void);
#endif // !TEST_NN
#include "cuTest.h"