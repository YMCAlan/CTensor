# CTensor
This is a tensor implementation in C.
Tensor is a Multidimensional Array commonly used in deep learning.
I am currently working on adding that feature.


Testing Framework:
* CuTest


## Tensor
The practical implementation is a 1d array, and element indexing is performed through a strided indexing scheme.

### Usage
```C
#include "src/tensor/tensor.h"
int shape[] = { 10 };
int img_shape[] = { 3,224,224 };
Tensor* input = createTensor(shape, 1, 1.0);
Tensor* image = createTensor(img_shape, 3, 0.0);
```

## To Do
* Decouple the method from the struct (Context).
* Implement computing graph (Too hard for me :rofl:

## Contributions
If you'd like to contribute to CTensor, please follow these steps:

1. Fork this repository to your GitHub account.
2. Make your changes in a branch.
3. Submit a pull request.

## License
This project is licensed under the LICENSE  - see the [LICENSE.md](https://github.com/YMCAlan/CTensor/blob/master/LICENSE.txt) file for details.
