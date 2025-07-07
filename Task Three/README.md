# Task Three
* Reference [Vision Transformer](https://github.com/kentaroy47/vision-transformers-cifar10)
* **torch.manual_seed(42)**: 
    When `torch.manual_seed(42)` is called, it initializes PyTorch's internal random number generator with the value 42. Subsequent calls to functions that rely on PyTorch's random number generation (e.g., `torch.randn`, `torch.rand`, weight initialization in neural networks) will produce the same sequence of "random" numbers each time the code is executed with that specific seed.
* While `torch.manual_seed()` is crucial for PyTorch-specific randomness, achieving full reproducibility in deep learning experiments often requires setting seeds for other libraries and functionalities as well:
    * **Python's random module**: `random.seed(seed_value)`
    * **NumPy**: `numpy.random.seed(seed_value)`
    * **CUDA (for GPU operations)**: `torch.cuda.manual_seed(seed_value)` for the current GPU, and `torch.cuda.manual_seed_all(seed_value)` for multi-GPU setups.
    * **Deterministic algorithms**: For certain CUDA operations, you might need to ensure deterministic behavior by setting `torch.backends.cudnn.deterministic = True` or `torch.use_deterministic_algorithms(True)`.