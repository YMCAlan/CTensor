#pragma once
#ifndef TEST_LINEAR_H
#define TEST_LINEAR_H
#include "cuTest.h"
#include "../src/nn/module.h"
#include "../src/linear/linear.h"
#include "../src/tensor/tensor.h"
void TestForwardLinear(CuTest* tc);
void TestBackwardLinear(CuTest* tc);
CuSuite* TestLinear(void);
#endif // !TEST_LINEAR_H
