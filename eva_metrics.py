from skimage.metrics import structural_similarity as ssim
import numpy as np
def snr(g,f,mode=1):
	"""
	SNR: calculate the signal-to-noise ratio (SNR)
	
	INPUT
	g: 		ground truth image
	f: 		noisy/restored image
	mode:	1->2D SNR, 2->3D SNR
	
	OUTPUT
	snr: 	SNR value
	
	The definition of SNR can be found in 
	Chen and Fomel, 2015, Random noise attenuation using local
	signal-and-noise orthogonalization, Geophysics.
	
	Author: Yangkang Chen, 2015
	"""
	
	import numpy as np

	if g.ndim==2:
		g=np.expand_dims(g, axis=2)

	if f.ndim==2:
		f=np.expand_dims(f, axis=2)
		
	g = np.double(g); #in case of data format is unit8,12,16
	f = np.double(f);

	if f.size != g.size:
		print('Dimesion of two images don''t match!');

	if mode ==1:
		s = g.shape[2];
		if s==1: #single channel	
			psnr = 20.*np.log10(np.linalg.norm(g[:,:,0],'fro')/np.linalg.norm(g[:,:,0]-f[:,:,0],'fro'));   
		else: #multi-channel
			psnr = np.zeros(s);
			for i in range(0,s):
				psnr[i] = 20.*np.log10(np.linalg.norm(g[:,:,i],'fro')/np.linalg.norm(g[:,:,i]-f[:,:,i],'fro'));

	else:
		[n1,n2,n3]=g.shape;
		psnr = 20.*np.log10(np.linalg.norm(g.reshape(n1,n2*n3,order='F'),'fro')/np.linalg.norm(g.reshape(n1,n2*n3,order='F')-f.reshape(n1,n2*n3,order='F'),'fro'));   

	return psnr

def mse_cal(g, f):
    if g.shape != f.shape:
        raise ValueError("The two cubes must have the same shape.")
    mse = np.mean((g - f)**2)
    return mse

def calculate_snr_psnr(cube1, cube2):
    """
    Compute SNR and PSNR between two 3D cubes.

    Parameters:
        cube1 (numpy.ndarray): Original 3D cube (signal).
        cube2 (numpy.ndarray): Reconstructed 3D cube.

    Returns:
        tuple: (SNR, PSNR) in dB.
    """
    if cube1.shape != cube2.shape:
        raise ValueError("The two cubes must have the same shape.")
    
    # Signal and noise
    signal_power = np.mean(cube1 ** 2)
    noise_power = np.mean((cube1 - cube2) ** 2)
    
    # SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    # Peak Signal Power (maximum pixel value)
    peak_signal_power = np.max(cube1) ** 2
    
    # PSNR
    psnr = 10 * np.log10(peak_signal_power / noise_power)
    
    return snr, psnr

def compute_3d_ssim(cube1, cube2, patch_size=11):
    """
    Compute SSIM for two 3D seismic cubes.

    Parameters:
        cube1 (numpy.ndarray): Original 3D seismic data (reference).
        cube2 (numpy.ndarray): Reconstructed 3D seismic data.
        patch_size (int): Size of the 3D patch for local SSIM computation.

    Returns:
        float: The average SSIM over all patches.
    """
    if cube1.shape != cube2.shape:
        raise ValueError("The two cubes must have the same shape.")
    
    ssim_map = np.zeros_like(cube1)
    total_ssim = 0.0
    count = 0
    
    # Iterate over the volume with sliding windows
    for z in range(0, cube1.shape[0] - patch_size + 1, patch_size):
        for x in range(0, cube1.shape[1] - patch_size + 1, patch_size):
            for y in range(0, cube1.shape[2] - patch_size + 1, patch_size):
                # Extract patches
                patch1 = cube1[z:z+patch_size, x:x+patch_size, y:y+patch_size]
                patch2 = cube2[z:z+patch_size, x:x+patch_size, y:y+patch_size]
                
                # Compute SSIM for the patch
                patch_ssim = ssim(patch1, patch2, data_range=cube1.max() - cube1.min())
                total_ssim += patch_ssim
                count += 1
    
    return total_ssim / count

def FrechetDistances(x,y):
    #Calculate mean and standard deviation on x
    mean_x , std_dev_x = np.mean(x), np.std(x)

    #Calculate mean and standard deviation on y
    mean_y , std_dev_y = np.mean(y), np.std(y)

    FD = np.sqrt((mean_x - mean_y) ** 2 + (std_dev_x - std_dev_y) ** 2)


    return FD