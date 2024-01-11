#pragma once
#ifndef TEST_TENSOR_H
#define TEST_TENSOR_H
#include "cuTest.h"


void TestCreateTensor(CuTest* tc);
void TestComputeStride(CuTest* tc);
void TestComputeSize(CuTest* tc);
void TestGetShape(CuTest* tc);
void TestGetSize(CuTest* tc);
void TestGetDim(CuTest* tc);
void TestIndexSetGet(CuTest* tc);
void TestReshape(CuTest* tc);
void TestPermute(CuTest* tc);
CuSuite* TestTensor(void);

#endif // !TEST_TENSOR_H
