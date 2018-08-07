# A Deep Introspection on Generative Adversarial Networks
## Abstract
GANs, namely Generative Adversarial Networks, are a hot topic nowadays. These models have the ability of generating good quality data, learning their distributions on a training set. Image generation in particular benefited from this framework, thanks also to the ease of assessment of the results obtained. Here we review the analysis of GAN models, recalling the necessary results from game theory, and propose a parametrization of the problem, which is then assessed analytically and by means of simulation, on image generation with MNIST dataset. With this parametrization it seems that in some cases it is possible to speed up the training phase.

# Models
### GAN Network
![gan_model](https://user-images.githubusercontent.com/31989563/43742790-97cb564e-99d3-11e8-8b09-51b6435eacdd.png)

### Discriminator Network
![d_model](https://user-images.githubusercontent.com/31989563/43742680-2b3db04e-99d3-11e8-9a7c-485777c31132.jpg)

### Generator Network
![g_model](https://user-images.githubusercontent.com/31989563/43742683-30a5eb3c-99d3-11e8-8641-1321cc245fc2.png)

# Results
### Results on MNIST
![mnist_digits_evoling](https://user-images.githubusercontent.com/31989563/43742869-c53102be-99d3-11e8-9240-6485fa83d46c.jpg)

### Results on 2D Distributions
![pdf_2d_fit](https://user-images.githubusercontent.com/31989563/43742691-3ddb00f8-99d3-11e8-9dbc-3e15d8bcc34b.png)

### Losses of 2D Distributions
![losses_2d-eps-converted-to](https://user-images.githubusercontent.com/31989563/43769753-c9dcbefe-9a3a-11e8-8d71-b5f6de24e5e9.jpg)

