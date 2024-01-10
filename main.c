#include<stdlib.h>
#include<stdio.h>

#include"src/cuTest/cuTest.h"

#include "src/tensor/tensor.h"
#include "src/linear/linear.h"
#include "src/nn/nn.h"

void RunAllTests(void) {
    // Create Suite
    CuString* output = CuStringNew();
    CuSuite* suite = CuSuiteNew();

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
    int new_img_shape[] = { 1, 3, 112, 112, 4, 1};
    image->reshape(image, new_img_shape, 6);
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