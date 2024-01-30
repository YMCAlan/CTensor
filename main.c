#include<stdlib.h>
#include<stdio.h>

#include "test/cuTest.h"
#include "test/testTensor.h"
#include "test/testLinear.h"
#include "test/testNN.h"
void RunAllTests(void) {
    // Create Suite
    CuString* output = CuStringNew();
    CuSuite* suite = CuSuiteNew();

    CuSuiteAddSuite(suite, TestTensor());
    CuSuiteAddSuite(suite, TestLinear());
    CuSuiteAddSuite(suite, TestNN());

    // Run the tests
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s\n", output->buffer);

    // Cleanup
    CuSuiteDelete(suite);
    CuStringDelete(output);
}

int main() {
    RunAllTests();
    return 0;
}