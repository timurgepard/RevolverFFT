# RevolverFFT

This is unfinished project called RevolverFFT.

One needs to add knowledge distallation. (Glimps of it was done in 2 model training together SLM in LLM repository: https://github.com/timurgepard/SLM_in_LLM)

The main Model is PicoFFT, FeedForward Transofrmer without Self-Attention (due to triangular mask gradients are stored only for 1/2 of matrix)

This version of PicoFFT was adjusted to work with RTX 3060 without depth.

Vocabular/Embedding Layer: 1440

Time Sequence: 1440

Facets (heads): 10

Layer: 1

Text Corpus is generated by ChatGPT to create Bitcoin ATM teller (one needs quantization to fit model to ATM)

![image](https://github.com/user-attachments/assets/1b0e7ca9-82cc-445f-9c57-d3a05cb2d38e)


![image](https://github.com/user-attachments/assets/7bc85fc1-2081-4307-8d8d-bfd34e440550)
