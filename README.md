# Image Captioning

To explore the fascinating intersections between computer vision and natural language processing, I implemented the image captioning model in [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) with [customizations](#Key-Info), which is a big improvement from [Show and Tell](https://arxiv.org/abs/1411.4555).

The encoder-decoder model proved their worth in machine translation tasks so researchers started using it for translating image features into language in a similar way. However, the biggest difference between translating the extracted features of an image and a French sentence both to an English sentence is that the visual information need to be heavily compressed into a just few bytes. Therefore, to build an image captioning model with the encoder-decoder model, the attention algorithm that is able to pick out only the key to higher performance. 

In this project, I learned a lot about integrating feature extraction with attention and LSTM, the underlying math equations from papers (best part of the paper), and even using PyTorch framework. Below is a sample result of my trained model. 

<p align="center"><img src="assets/results/1.jpg"></p>

---

## Background

From [Show, Attend and Tell](https://arxiv.org/abs/1502.03044):
> Automatically generating captions of an image is a task very close to the heart of scene understanding - one of the primary goals of computer vision.

Neural image captioning is about giving machines the ability of compressing salient visual information into descriptive language. The biggest challenges are building the bridge between computer vision and natural language processing models and producing captions that described the most significant aspects of the image.

<p align="center"><img src="assets/architecture.png" width="75%" height="75%"></p>

For detailed background info on feature extraction, soft/hard self-attention, and sequence generation with LSTM, [resources section](#Resources) contains a number of useful links/papers I used. Wrapping your head around how image encoding, attention, and LSTM come together is the key to understanding this implementation (top-down approach). I watched videos and read blogs to get the overall architecture then dived into the [the paper](https://arxiv.org/abs/1502.03044) to understand the math formulae. Additionally, the comments on my code might help. 

---

## Key Info

Below are some of my choices about the implementation (chronological order).

- **PyTorch** both for its pythonic syntax and to utilize the strong GPU acceleration. There is less documentation on PyTorch so I ended up learning a lot more by reading a bit of source code.
- **Colab's T4 GPU** from google (thank you!), which was strong enough for Flickr30k with small batch sizes (max 12).
- **Flickr30k** dataset because MS COCO requires enormous training time and computational power for Colab. [Link to download](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Andrej Karpathy.
- **No pre-trained embedding** because training my own embedding is not so computationally expensive and fits to context
- **Soft attention** (deterministic) for its differentiability (simple standard backprop). Intuitively, soft attention looks at the whole image while focusing on some parts while hard attention only looks at one randomly weighted choice at a time.
- **Mult-layer perceptron** for the attention model, as from [the paper](https://arxiv.org/abs/1502.03044).
- **Doubly stochastic attention regularization parameter** was used to encourage the model to pay equal attention to every part of the image over a course of generation. This was used to improve the score in [the paper](https://arxiv.org/abs/1502.03044).
- **BLEU-4** score for both training (early stopping) and evaluation.
- **Teacher forcing** (use GT as input to each LSTMCell iteration) to achieve faster convergence. `Equation 6` of [the paper](https://arxiv.org/abs/1502.03044) clearly indicates the dependence of the context vector (attention output) on the previous hidden state, which itself is dependent on the previous outputs of the LSTM decoder (trace back to `equations 1, 2, 3`) .mlp 
- **Beam Search** to find the most optimal sequence after decoder does the heavy lifting.

---

## Performance Evaluation

Under development.

---

## Try it Yourself

#### Dependencies

Under development.

#### Train

Under development.

---

## Possible Improvements

- Better hardware enablese MS COCO, higher batch size, higher epoch -> higher performance
- Try hard-attention and compare performances
- Fine-tune ResNet longer to fit the dataset
- Perform image augmentation (ie. horizontal flip) ie MS COCO is still not enough (unlikely)
- Instead of constant teacher forcing, [scheduled sampling](https://arxiv.org/pdf/1506.03099.pdf) has been proven to be better based on probability.
- As mentioned in [the paper](https://arxiv.org/abs/1502.03044), a major drawback of using attention is distilling the important parts of an image especially on images that have a lot of things going on. This problem is addressed by [DenseCap](https://cs.stanford.edu/people/karpathy/densecap/) where the objects are first recognized in separate windows. 

---

## Resources

#### Papers

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- [Fully Convolutional Localization Networks for Dense Captioning](https://cs.stanford.edu/people/karpathy/densecap/)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/pdf/1506.03099.pdf)

#### Miscellaneous

- [Pytorch documentation](https://pytorch.org/docs/stable/index.html)
- [Colab documentation](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true)
- [List of related repositories](https://github.com/zhjohnchan/awesome-image-captioning)
- [Blog Tutorial on Image captioning without attention](https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/)
- [Github Tutorial on image captioning with attention](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#caption-lengths)
- [Attention video explanation + a bit of image captioning](https://www.youtube.com/watch?v=W2rWgXJBZhU)