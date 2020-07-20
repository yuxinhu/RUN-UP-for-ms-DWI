# RUN-UP-for-ms-DWI

### Introduction
An [unrolled pipeline](https://arxiv.org/abs/1705.08041) containing recurrences of model-based gradient updates and neural networks is introduced with several new features for the reconstruction of multi-shot DWI. The network is trained to predict the results of jointly-reconstructed multi-direction data from [SPA-LLR](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28025) while still using single-direction data as input. The generalizability of the proposed method is also demonstrated in the breast data. 

In this method, we alternate the input space between k-space and image space which allows the network to directly fill unacquired data in the k-space domain, then refine and denoise the results in the image domain. The proposed method shows a robust reconstruction with only six iterations. In addition, we feed the non-diffusion-weighted image to the network to utilize its shared structure similarities to the image to be reconstructed. The usage of [parametric ReLU](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) instead of Leaky ReLU also helps reduce the bias in the k-space domain. Moreover, we use an intermediate loss, which depends on the results from early iterations to help train this deep CNN. For more details, please refer to [our RUN-UP paper]() published on MRM.

### Explanations 
There are mainly three parts of code:

(1) Data preparation: (i) including k-space data extraction, (ii) image reconstruction of different methods, (iii) data preparation for different methods, and different cases, (iv) saving all data into a folder (one file for each slice, direction, and nex). Ther k-space data reading (step_1i_demo_DL_readP.m) is based on the modified Matlab Orchestra function (EpiDiffusionRecon_yuxin.m). The reconstruction of the k-space data involves two methods (1) SPA-LLR as the training target (a demo could be found [here](https://github.com/yuxinhu/SPA-LLR)), and (2) shot-LLR for comparison (step_1ii_demo_DL_shot_LLR.m, shot-LLR serves as target for breast DWI). [BART](https://mrirecon.github.io/bart/) is needed for shot-LLR reconstruction. For each diffusion-encoding direction and slice, we are going to save the corresponding k-space data, sensitivity map, and some other information and shot-LLR reconstruction results into one file. So that with very light processing, this could be used as the training data of the deep learning reconstruction. The idea and the implementation of steps (i) and (ii) are pretty similar to that of the [SPA-LLR reconstruction](https://github.com/yuxinhu/SPA-LLR), please refer to the explanations there. For step (iii), if we want to include b=0 images as the input to the reconstruction network, we need to first save a b0.mat file which contains b=0 images of all slices for each case (step_1iii_demo_DL_breast_save_b0.m for breast dat or step_1iii_demo_DL_nonc_save_b0_gt.m for DTI data or step_1iii_demo_DL_6nex_save_b0.m for 6-nex data); if we want to use joint reconstruction results as the ground truth, we need to reformat the joint reconstruction results as in step_1iii_demo_DL_nonc_save_b0_gt.m). Then we can construct the training data as in step_1iv_demo_DL_savecfl_b0.m or step_1iv_demo_DL_savecfls.m for step (iv). Notice the format of the training data is defined in the function "save_cfl", and the saved cfl files can be loaded by function "DL_loadcfl". Finally, after these three steps, we should be able to get a folder containing a list of files, and each file contains the k-space data, sensitivity map, target results, and other related information. I have to say this is a very complicated process, since you need to read the data, do the reconstruction, and reformat the results. A lot can be done to optimize this process to make it more robust.

(2) Training, fine-tuning, and test of the network (run_up.py): this part is based on some starter code from [Dr. Joseph Cheng](http://mrsrl.stanford.edu/~jycheng/) and [Dr. Xinwei Shi](http://stanford.edu/~xinweis/). Our strategy is to save one file for one example (we treat one slice as one example). For each file, it saves the k-space data, sensitivity map, sampling operator, and the ground truth (from SPA-LLR) into a long vector (this sounds weird, but it works well otherwise there might be some bugs. Hopefully latest TensorFlow does not have this problem). You can use your own way, by changing the data preparation and data loading part. Another thing is that we did not use data augmentation, not even shift. This is because even with the easiest shift, we can not make the result correct. I could not remember the error/problem exactly but it was really annoying. At the beginning, we thought it was due to the half-pixel shift or something like that, but it turns out to be wrong. I think this might be something due to TensorFlow. So unless you project is very similar to this one, I would suggest you use some other framework to reimplement the whole network, which may be much cleaner and easier to maintain, especially for the unrolled network/U-Net part.

(3) Post-processing the data: including reorganization and evaluation of the results (demo_DL_save_com_res.m). This part should be pretty straightforward.

### Usage
All data are saved on our group server, please refer to [this sheet](). One trained network is saved on our group server /bmrNAS/people/yuxinh/DL_data_revision/model_saved/nonc_iter6_ki_32filter_interloss/. It is trained with the following command (you probably need to set the paths yourselves):

        python run_up.py --mode="train" --input_dir=“TRAIN_DATA_PATH” --output_dir=“OUTPUT_PATH” --net="unet" --num_filters=32 --num_steps=2 --num_unroll_iter=6 --batch_size=3 --unroll_opt="fista" --no_do_hardproj  --do_residual_conv --stride_y=4 --lr=0.0004 --intermediate_loss --addb0=1 --lrelu=0

The fine-tuning command:

        python run_up.py --mode="train" --input_dir=“TRAIN_DATA_PATH” --output_dir=“OUTPUT_PATH” --checkpoint=“TRAINED_NETWORK_PATH” --net="unet" --num_filters=32 --num_steps=2 --num_unroll_iter=6 --batch_size=3 --unroll_opt="fista" --no_do_hardproj  --do_residual_conv --stride_y=4 --lr=0.0004 --intermediate_loss --addb0=1 --lrelu=0

The test command:

        python run_up.py --mode=“test” --input_dir=“TEST_DATA_PATH” --output_dir=“OUTPUT_PATH”  --checkpoint=“TRAINED_NETWORK_PATH” --net="unet" --num_filters=32 --num_steps=2 --num_unroll_iter=6 --batch_size=3 --unroll_opt="fista" --no_do_hardproj  --do_residual_conv --stride_y=4 --lr=0.0004 --intermediate_loss --addb0=1 --lrelu=0

Some notes about visualization using Tensorboard which may be super helpful:

    You can port-forward with another ssh command that need not be tied to how you are connecting to the server (as an alternative to the other answer). Thus, the ordering of the below steps is arbitrary.
    
        1. from your local machine, run
                ssh -N -f -L localhost:16006:localhost:6006 yuxinh@roma
    
        2. on the remote machine, run tensorboard --logdir <path>
                (-N : no remote commands
                 -f : put ssh in the background 
                 -L <machine1>:<portA>:<machine2>:<portB> : forward <machine2>:<portB> (remote scope) to <machine1>:<portA> (local scope)
  
     Then from you local machine, go to http://127.0.0.1:16006 and then you should be able to check the real-time results.













