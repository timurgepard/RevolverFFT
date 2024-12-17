# RevolverFFT

This is unfinished project called RevolverFFT.

One needs to add knowledge distallation. (Glimps of it was done in 2 model training together SLM in LLM repository)

The main Model is PicoFFT, FeedForward Transofrmer without Self-Attention (due to triangular mask gradients are stored only for 1/2 of matrix)

This version of PicoFFT was adjusted to work with RTX 3060 without depth.

Vocabular/Embedding Layer: 1440
Time Sequence: 1440
Facets (heads): 10
Layer: 1
