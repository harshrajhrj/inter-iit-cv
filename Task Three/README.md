# Task Three
* Reference [Vision Transformer](https://github.com/kentaroy47/vision-transformers-cifar10)
* [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
* [A Deep Dive into the Self-Attention Mechanism of Transformers](https://medium.com/analytics-vidhya/a-deep-dive-into-the-self-attention-mechanism-of-transformers-fe943c77e654)
* [Understand the implementation](https://youtu.be/7o1jpvapaT0?si=e1I6VEXtlSpFgfSZ)
* **torch.manual_seed(42)**: 
    When `torch.manual_seed(42)` is called, it initializes PyTorch's internal random number generator with the value 42. Subsequent calls to functions that rely on PyTorch's random number generation (e.g., `torch.randn`, `torch.rand`, weight initialization in neural networks) will produce the same sequence of "random" numbers each time the code is executed with that specific seed.
* While `torch.manual_seed()` is crucial for PyTorch-specific randomness, achieving full reproducibility in deep learning experiments often requires setting seeds for other libraries and functionalities as well:
    * **Python's random module**: `random.seed(seed_value)`
    * **NumPy**: `numpy.random.seed(seed_value)`
    * **CUDA (for GPU operations)**: `torch.cuda.manual_seed(seed_value)` for the current GPU, and `torch.cuda.manual_seed_all(seed_value)` for multi-GPU setups.
    * **Deterministic algorithms**: For certain CUDA operations, you might need to ensure deterministic behavior by setting `torch.backends.cudnn.deterministic = True` or `torch.use_deterministic_algorithms(True)`.

### Observations
**Training Configuration 1**
```py
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1
```
* Training Accuracy: $76.212$%
* Test Accuracy: $63.19$%


![Accuracy Plot](./assets/output_1.png)
![Prediction Plot](./assets//output_2.png)
---

**Training Configuration 2**
```py
BATCH_SIZE = 128
EPOCHS = 20 # applied change
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 16 # applied change
DEPTH = 6
MLP_DIM = 512
DROP_RATE = 0.1
```
* Training Accuracy: $93.888$%
* Test Accuracy: $63.29$%


![Accuracy Plot](./assets/output_3.png)
![Prediction Plot](./assets/output_4.png)