## Automatic Differentiation and CNNs

This project is heavily inspired by [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/geohot/tinygrad).

In this project I wrote an AD library and put it to the test by training a CNN on the MNIST dataset.

In the folder `AutomaticDifferntiation` you can find the core types for scalars, vectors, matrices and tensors as well as the logic for backpropagating basic functions such as addition, multiplication, subsetting, etc.

In `ADNN` I use these types and implement a convolutional neural net. This contains dense, convolutional and maxpool layers. Of course, also activations and losses are there.

In `mnist/mnist.jl` I tried three different models on the MNIST dataset. The last one, consisting with the architecture
```
  # First convolution, operating upon a 28x28 image
  Conv((3, 3), 1=>16, relu),
  MaxPool((2,2)),

  # Second convolution, operating upon a 13x13 image
  Conv((3, 3), 16=>32, relu),
  MaxPool((2,2)),

  # Third convolution, operating upon a 5x5 image
  Conv((3, 3), 32=>32, relu),
  MaxPool((2,2)),

  # Reshape 3d tensor into a 2d one using flatten, at this point it should be (1, 1, 32, N)
  flatten,
  Dense(32, 10)
```

reached 97.6 % accuracy after 20 epochs which takes around 26 minutes on my Macbook Pro 2017 i5. In comparison, an implementation of the same model in Flux.jl takes 18 minutes to train. But the goal of this project was not performance.
