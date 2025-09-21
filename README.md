# SeisReconNO: Leveraging a U-Net-Enhanced Fourier Neural Operator for 3D Seismic Reconstruction
<div align="center">
  <a href="https://github.com/traversa942" target="_blank">Alessandro Traversa<sup>1</sup></a> &emsp;
  <a href="https://github.com/cuiyang512" target="_blank">Yang Cui<sup>1</sup></a> &emsp;
  <a href="https://cpg.kfupm.edu.sa/bio/dr-umair-bin-waheed/" target="_blank">Umair bin Waheed<sup>1</sup></sup></a> &emsp;
  <a href="https://www.jsg.utexas.edu/researcher/yangkang_chen" target="_blank">Yangkang Chen<sup>2</sup></a>
</div>
<div align="center">
  <sup>1</sup> College of Petroleum Engineering & Geosciences, King Fahd University of Petroleum & Minerals <br>
  <sup>2</sup> Jackson School of Geosciences, University of Texas at Austin
</div>


## Abstract
Missing traces in 3D seismic data are a recurring challenge caused by receiver malfunctions, acquisition limitations, and geological or environmental constraints. These gaps hinder accurate interpretation and further processing. Although numerous model-driven approaches have been developed in recent decades, they often struggle with reconstructing the data with complex geological structures and high missing ratios. To address these limitations, we proposed a U\textendash Net-enhanced Fourier Neural Operator (UFNO) 3D seismic reconstruction framework to achieve a mesh-invariant seismic reconstruction across different missing scenarios. The UFNO model leverages both spectral and spatial representations to learn a generalized reconstruction operator. We train the model on field 3D seismic cubes featuring three key missing-data patterns: random, trace-wise,  and regular. Experimental results demonstrate the superior reconstruction capability of UFNO across varying missing ratios. Moreover, the model exhibits strong generalization to unseen data with different resolutions, confirming its potential as a robust and adaptable tool for seismic data enhancement in real-world applications.


## Install 
For set up the environment and install the dependency packages, please run the following script:
    
    conda create -n Seisrecon python=3.12
    conda activate Seisrecon
    conda install ipython notebook
    pip install torch==2.5.1, numpy==2.2.0, matplotlib==3.10.0, scikit-image==0.25.0, cigsegy==1.1.8, pylops==2.4.0


    ## Model and data

Regarding the training data, we used two open source post-stack 3D seismic data:

  1) **F3 data from Netherland:** [F3 open-source data](https://wiki.seg.org/wiki/F3_Netherlands)
  2) **Moomba lake data from Austrilia:** [Moomba open-source data](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/wci/ResultSet?w=NATIVE(%27REFERENCE+ph+is+%22Env%2009124%22%27);order=TITLE;r=1;m=1;rpp=25)
  3) **Kerry data from New Zealand:** [Kerry open-source data](https://wiki.seg.org/wiki/Kerry-3D)
  4) **Due to the limitation of Github, we uploaded all the saved training model to a Google Drive folder. Please download them from the attached link:** [save wieght matrix Goole Drive](https://drive.google.com/drive/folders/1CjeKiHW0GV_PfeqWZDzzwb2J0Zm6tEA1?usp=sharing)

After downloading the given files, please put the trained model into the "model" folder.

The model of Wen et al. 2022 was modified for seismic reconstruction purposes. The first layer, through which the data pass, is the padding layer, which consists of adding additional values (zeros, constants or other specified values) around the edges of the tensor to preserve edge information in the convolutional layers. After that, two Fourier neural operator (FNO) layers are applied. These FNO layers use the Fourier transform to obtain features from the frequency domain, but it also incorporates operations in the spatial domain to complement this and refine the extracted them. The workflow is as follows:
     
   - **Fourier Transform:** The input tensor is converted from the spatial domain to the frequency domain using the fast Fourier transform (FFT). Since we are working with three dimensions in the neural network (timeslice, inline and xline).
   -  **Learnable Weight Multiplication:** The Fourier coefficients are modified by multiplying them by learnable weights.
   -  **Truncation of Frequency Modes:** To avoid over-fitting, only a fixed number of low-frequency modes are retained and this hyper is called modes, that for this model is 10.
   -  **Inverse Fourier Transform:** The modified Fourier coefficients are transformed back to the spatial domain using the Inverse FFT.
   -  **Spatial Operations in FNO:** The FNO layer uses additional linear transformations directly in the spatial domain to complement the global features extracted in the Fourier domain by adding them, the widht of this linear transformatons are 16.

After the FNO layer, two more U-FNO layers are added. They are similar to the FNO, but a U-net is added to the model in the spatial domian representation. In general, the functionality of the U-net is downsampling the data by passing through successive convolutional layers and pooling or strided convolutions to reduce spatial dimensions, extract representation in a latent space and after upsampling the data. This addition of U-Net-like convolutional layers increases the extraction features. Among the hyperparameters that we used are a dropuout system, a kernel size of 3, Leaky ReLU was used as activation function. The  architecture used was thre downsampling convolutional layers and upsampling convolutional layers.

At the end of the model, the original dimensions are restored by removing the extra padding using slicing

![Training model workflow](figs/UFNO_architecture.png)
