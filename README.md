# RUN-UP-for-ms-DWI


### Introduction

We introduced unrolled network with U-nets as priors. Similar to conventional assumption-based (e.g., sparsity, low-rank, etc.) reconstructions, we solve the problem iteratively. And in each step, we have one gradient update based on MRI acquisition mode. The difference is that after the gradient update, we update the image based on the trained network instead of the assumptions. The network is trained on previously acquired and reconstructed data. It is showed that the use of network can significantly reduce the required number of iteartions, thus accelerate image reconstruction.

One difficulty of doing mult-shot DWI reconstruction using deep learning is about how to get ground truth. We use SPA-LLR, in which multi-direction data are jointly reconstructed, as ground truth. This way, the trained network is not only faster than conventional methods, which reconstrct each direction independently, but also can provide better results.
### Usage 

The implementation is based on some starter code from our lab (specailly thanks to Dr. Xinwei Shi).

Training data preparation:

We could upload our data here since they are too large and there might be some privacy issues. Our strategy is to save one file for one example (we treat one slice as one example). For each file, it contains the k-space data, sensitivity map, sampling operator, and the ground truth (from SPA-LLR). You can use your own way, by chaning function "" about loading data.






