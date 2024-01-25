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
Tensor* input = createTensor(SHAPE(10), 1, 1.0);
Tensor* image = createTensor(SHAPE(3,224,224) , 3, 0.0);

Linear* linear = createLinear(10, 20, true);
```

## To Do
* Decouple the method from the struct.
* Implement Convoulation. 📝
* Implement Network container (may be generic. 📝

## Contributions
If you'd like to contribute to CTensor, please follow these steps:

1. Fork this repository to your GitHub account.
2. Make your changes in a branch.
3. Submit a pull request.

## License
This project is licensed under the LICENSE  - see the [LICENSE.md](https://github.com/YMCAlan/CTensor/blob/master/LICENSE.txt) file for details.
