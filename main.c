#include<stdlib.h>
#include<stdio.h>

#include"testStack.h"
#include"testQueue.h"
#include"testPair.h"
#include"cuTest.h"

#include "tensor.h"
#include "linear.h"
#include "nn.h"

void RunAllTests(void) {
    // Create Suite
    CuString* output = CuStringNew();
    CuSuite* suite = CuSuiteNew();
    CuSuiteAddSuite(suite, TestStack());
    CuSuiteAddSuite(suite, TestQueue());
    CuSuiteAddSuite(suite, TestPair());

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
    // create vector
    int shape[] = { 10 };
    int img_shape[] = { 3,224,224 };
    Tensor* input = createTensor(shape, 1, 1.0);
    Tensor* image = createTensor(img_shape, 3, 0.0);
    NN* nn = createNN();

    nn->addLayer(nn, createLinear(10, 20, false));
    nn->addLayer(nn, createLinear(20, 30, false));
    nn->addLayer(nn, createLinear(30, 2, false));

    Tensor* output = nn->nnForward(nn, input);
    output->print(output);

    int dim = output->dim;
    int* output_shape = output->shape;
    for (int i = 0; i < dim; i++)
    {
        printf("Shape = %d", output_shape[i]);
    }

    return 0;
}