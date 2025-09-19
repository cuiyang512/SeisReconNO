import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap



def irregular2(nry, nrx, nt, perc_sub=0.4, seed=10):
    """Random irregular sampling along two directions

    Create indices to perform random irregular sampling
    along two directions

    Parameters
    ----------
    nry : :obj:`int`
        Number of receivers along y axis
    nrx : :obj:`int`
        Number of receivers along x axis
    nt : :obj:`int`
        Number of time samples of data to which subsampling will be applied
    perc_sub : :obj:`float`, optional
        Percentage of subsampling
    seed : :obj:`int`, optional
        Random seed

    Returns
    -------
    iava : :obj:`numpy.ndarray`
        Selected indices (including time axis) to be used to decimate the data
    iavarec : :obj:`numpy.ndarray`
        Selected indices along receiver grid

    """
    np.random.seed(seed)
    nsub = int(np.round(nry * nrx * perc_sub))
    iavarec = np.sort(np.random.permutation(np.arange(nry * nrx))[:nsub]).astype(int)

    # create mask
    mask = np.zeros((nry * nrx, nt))
    mask[iavarec, :] = 1
    mask = mask.reshape(nry, nrx, nt)
    iava = np.where(mask.ravel() == 1)[0]
    return iava, iavarec

def dithered_irregular2(nry, nrx, nt, factor_sub=3, seed=10):
    """Dithered irregular sampling along two directions

    Create indices to perform dithered irregular sampling
    along two directions

    Parameters
    ----------
    nry : :obj:`int`
        Number of receivers along y axis
    nrx : :obj:`int`
        Number of receivers along x axis
    nt : :obj:`int`
        Number of time samples of data to which subsampling will be applied
    factor_sub : :obj:`int`, optional
        Factor of subsampling
    seed : :obj:`int`, optional
        Random seed

    Returns
    -------
    iava : :obj:`numpy.ndarray`
        Selected indices (including time axis) to be used to decimate the data
    iava2d : :obj:`numpy.ndarray`
        Selected indices along receiver grid
    iava2d_reg : :obj:`numpy.ndarray`
        Indices of regularly subsampled grid (prior to dithering) along receiver grid
    dither : :obj:`numpy.ndarray`
        Dithers

    """
    np.random.seed(seed)
    iava_regy, iava_regx = np.arange(nry)[::factor_sub], np.arange(nrx)[::factor_sub]

    iava_regy, iava_regx = np.meshgrid(iava_regy, iava_regx, indexing='ij')
    iava_reg = np.vstack((iava_regy.ravel(), iava_regx.ravel()))
    nr_reg = iava_reg.shape[1]

    # create dither code
    dithery = np.random.randint(-factor_sub // 2 + 1, factor_sub // 2 + 1, nr_reg)
    ditherx = np.random.randint(-factor_sub // 2 + 1, factor_sub // 2 + 1, nr_reg)
    dither = np.vstack((dithery.ravel(), ditherx.ravel()))

    # Apply dithers
    iavarec = iava_reg + dither
    
    # Clamp indices to stay within grid bounds
    iavarec[0] = np.clip(iavarec[0], 0, nry - 1)
    iavarec[1] = np.clip(iavarec[1], 0, nrx - 1)
    
    # Create 2D mask
    mask = np.zeros((nry, nrx))
    mask[iavarec[0], iavarec[1]] = 1
    mask = mask.reshape(nry, nrx)
    iava2d_reg = np.where(mask.ravel() == 1)[0]

    # create 2d mask
    mask = np.zeros((nry, nrx))
    mask[iavarec[0], iavarec[1]] = 1
    mask = mask.reshape(nry, nrx)
    iava2d = np.where(mask.ravel() == 1)[0]

    # create mask
    mask = np.zeros((nry, nrx, nt))
    mask[iavarec[0], iavarec[1], :] = 1
    iava = np.where(mask.ravel() == 1)[0]
    return iava, iava2d, iava2d_reg, dither


def genmask(u, r, type, seed):
	"""
	GENMASK:Generate Random Sampling Mask
	
	INPUT
	u: 		image
	r: 		data KNOWN ratio
	type: 	data lose type
	   		'r': random lose rows
	   		'c': random lose columns
	   		'p': random lose pixel
	seed: 	seed of random number generator
	
	OUTPUT
	mask: 	sampling mask
	
	  Copyright (C) 2014 The University of Texas at Austin
	  Copyright (C) 2014 Yangkang Chen
	  Ported 
	
	  This program is free software: you can redistribute it and/or modify
	  it under the terms of the GNU General Public License as published
	  by the Free Software Foundation, either version 3 of the License, or
	  any later version.
	
	  This program is distributed in the hope that it will be useful,
	  but WITHOUT ANY WARRANTY; without even the implied warranty of
	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	  GNU General Public License for more details:
	  http://www.gnu.org/licenses/
	"""
	
	m=u.shape[0];
	n=u.shape[1];
	

	mask = np.zeros([m,n]);
	
	if type=='r':
		row = rperm(m,seed);
		k = np.fix(r*m);k=int(k);
		row = row[0:k-1];
		mask[row,:] = 1;
		
	elif type=='c':
		column = rperm(n,seed);
		k = np.fix(r*n);k=int(k);
		column = column[0:k-1];
		mask[:, column] = 1;
	
	elif type=='p':
		pix = rperm(m*n,seed);
		k = np.fix(r*m*n);k=int(k);
		pix = pix[0:k-1];
		mask[pix]= 1;
	else:
		print("mask type not found");
		
	return mask

def rperm(n,seed):
	"""
	RPERM: Random permutation of my version.
	
	RPERM(n) is a random permutation of the integers from 1 to n.
	For example, RANDPERM(6) might be [2 4 5 6 1 3].
	
	  Copyright (C) 2014 The University of Texas at Austin
	  Copyright (C) 2014 Yangkang Chen
	
	  This program is free software: you can redistribute it and/or modify
	  it under the terms of the GNU General Public License as published
	  by the Free Software Foundation, either version 3 of the License, or
	  any later version.
	
	  This program is distributed in the hope that it will be useful,
	  but WITHOUT ANY WARRANTY; without even the implied warranty of
	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	  GNU General Public License for more details:
	  http://www.gnu.org/licenses/
	"""
	
	np.random.seed(seed);
	p = np.argsort(np.random.rand(n));

	return p

def large_gap_missing_3d(data, direction='inline', section_num=3, trace_each_gap=10, rand_seed=202411):
    """
    Generate large gap missing traces in a 3D seismic cube.

    Author: Yang Cui
    Email: yang.cui512@gmail.com
    Date: Nov. 2024

    INPUT
        data: 3D seismic cube (dimensions: inline, xline, time)
        direction: 'inline' or 'crossline', specifies the direction of gaps
        section_num: number of missing sections to create
        trace_each_gap: number of traces in each missing section
        rand_seed: random seed for reproducibility

    OUTPUT
        data: Modified seismic cube with gaps
    """

    # Seismic cube dimensions
    seis_data = np.copy(data)
    n_ilines, n_xlines, n_samples = seis_data.shape

    gap = trace_each_gap
    num_sections = section_num
    random.seed(rand_seed)  

    # Keep track of already selected ranges to avoid overlap
    over_laps_range = []

    if direction == 'inline':
        # print('Zeroing out along the inline direction')
        for _ in range(num_sections):
            while True:
                # Choose a random starting xline index
                start_xline = random.randint(0, n_xlines - gap)
                
                # Check for overlap with existing ranges
                if all(start_xline + gap <= r[0] or start_xline >= r[1] for r in over_laps_range):
                    break  # Found a valid range

            # Zero out the specified range along the inline direction
            seis_data[:, start_xline:start_xline + gap, :] = 0

            # Record the range to avoid overlap
            over_laps_range.append((start_xline, start_xline + gap))

    elif direction == 'crossline':
        # print('Zeroing out along the crossline direction')
        for _ in range(num_sections):
            while True:
                # Choose a random starting inline index
                start_iline = random.randint(0, n_ilines - gap)
                
                # Check for overlap with existing ranges
                if all(start_iline + gap <= r[0] or start_iline >= r[1] for r in over_laps_range):
                    break  # Found a valid range

            # Zero out the specified range along the crossline direction
            seis_data[start_iline:start_iline + gap, :, :] = 0

            # Record the range to avoid overlap
            over_laps_range.append((start_iline, start_iline + gap))

    else:
        print('Please choose a valid direction: "inline" or "crossline"')
        return seis_data  # Return the original data without modification

    # print("Zeroing process completed.")
    return seis_data



def data_norm(seis_data):
    '''
    author: Yang Cui
    email: yang.cui512@gmail.com
    date: Nov. 2024
    '''
    norm_data = []
    data = np.array(seis_data)
    for ii in range(0, data.shape[0]):
        slice_2d = data[ii, :, :]
        sclip = np.abs(np.percentile(slice_2d, 0.999))
        xmax, xmin = sclip, -sclip
        slice_2d = (slice_2d - xmin)/(xmax - xmin)
        norm_data.append(slice_2d)
    norm_data = np.array(norm_data)
    
    norm_data = (data - data.mean())/data.std()
    norm_data = norm_data.astype('float32')
    print(f"min : {norm_data.min()}, max : {norm_data.max()}, mean : {norm_data.mean()}, std : {norm_data.std()}")
    return norm_data 

import numpy as np

def regular_missing(cube_data, n, mode="inline"):
    """
    Apply regular missing traces to a 3D seismic cube.
    
    Args:
        cube_data : np.ndarray
            Input cube of shape (nx, ny, nz).
        n : int
            Keep every n-th trace (missing ratio = (n-1)/n).
        mode : str
            "inline"     -> regular missing along axis 0
            "crossline"  -> regular missing along axis 1
            "both"       -> inline + crossline
            "all"        -> inline + crossline + vertical (axis 2)
    Returns:
        masked_cube : np.ndarray
            Cube with missing traces applied.
    """
    cube_data = cube_data.T  # match your input convention
    in1, in2, in3 = cube_data.shape
    mask = np.ones((in1, in2, in3))

    if mode in ["inline", "both", "all"]:
        offset = np.random.randint(0, n)
        retained = list(range(offset, in1, n))
        mask = mask * 0
        mask[retained, :, :] = 1 if mode == "inline" else mask[retained, :, :]

    if mode in ["crossline", "both", "all"]:
        offset = np.random.randint(0, n)
        retained = list(range(offset, in2, n))
        if mode == "crossline":
            mask = np.zeros_like(cube_data)
        mask[:, retained, :] = 1 if mode == "crossline" else mask[:, retained, :]

    if mode == "all":
        offset = np.random.randint(0, n)
        retained = list(range(offset, in3, n))
        mask[:, :, :] = 0
        mask[:, :, retained] = 1

    masked_cube = (cube_data * mask).T
    return masked_cube

def cseis():
	'''
	cseis: seismic colormap
	
	By Yangkang Chen
	June, 2022
	
	EXAMPLE
	from pyseistr import cseis
	import numpy as np
	from matplotlib import pyplot as plt
	plt.imshow(np.random.randn(100,100),cmap=cseis())
	plt.show()
	'''
	seis=np.concatenate(
(np.concatenate((0.5*np.ones([1,40]),np.expand_dims(np.linspace(0.5,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((0.25*np.ones([1,40]),np.expand_dims(np.linspace(0.25,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((np.zeros([1,40]),np.expand_dims(np.linspace(0,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose()),axis=1)

	return ListedColormap(seis)

def plot3d(d3d,frames=None,z=None,x=None,y=None,dz=0.01,dx=0.01,dy=0.01,nlevel=100,figsize=(8, 6),figname=None,showf=True,close=True,**kwargs):
	'''
	plot3d: plot beautiful 3D slices
	
	INPUT
	d3d: input 3D data (z in first-axis, x in second-axis, y in third-axis)
	frames: plotting slices on three sides (default: [nz/2,nx/2,ny/2])
	z,x,y: axis vectors  (default: 0.01*[np.arange(nz),np.arange(nx),np.arange(ny)])
	figname: figure name to be saved (default: None)
	showf: if show the figure (default: True)
	close: if not show a figure, if close the figure (default: True)
	kwargs: other specs for plotting
	dz,dx,dy: interval (default: 0.01)
	
	By Yangkang Chen
	June, 18, 2023
	
	EXAMPLE 1
	import numpy as np
	d3d=np.random.rand(100,100,100);
	from pyseistr import plot3d
	plot3d(d3d);
	
	EXAMPLE 2
	import scipy
	data=scipy.io.loadmat('/Users/chenyk/chenyk/matlibcyk/test/hyper3d.mat')['cmp']
	from pyseistr import plot3d
	plot3d(data);
	
	EXAMPLE 3
	import numpy as np
	import matplotlib.pyplot as plt
	from pyseistr import plot3d

	nz=81
	nx=81
	ny=81
	dz=20
	dx=20
	dy=20
	nt=1501
	dt=0.001

	v=np.arange(nz)*20*1.2+1500;
	vel=np.zeros([nz,nx,ny]);
	for ii in range(nx):
		for jj in range(ny):
			vel[:,ii,jj]=v;

	plot3d(vel,figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dz,y=np.arange(nz)*dz,barlabel='Velocity (m/s)',showf=False,close=False)
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('3D velocity model')
	plt.savefig(fname='vel3d.png',format='png',dpi=300)
	plt.show()
	
	'''

	[nz,nx,ny] = d3d.shape;
	
	if frames is None:
		frames=[int(nz/2),int(nx/2),int(ny/2)]
		
	if z is None:
		z=np.arange(nz)*dz
	
	if x is None:
		x=np.arange(nx)*dx
		
	if y is None:
		y=np.arange(ny)*dy
	
	X, Y, Z = np.meshgrid(x, y, z)
	
	d3d=d3d.transpose([1,2,0])
	
	
	kw = {
	'vmin': d3d.min(),
	'vmax': d3d.max(),
	'levels': np.linspace(d3d.min(), d3d.max(), nlevel),
	'cmap':'seismic'
	}
	
	kw.update(kwargs)
	
	if 'alpha' not in kw.keys():
		kw['alpha']=1.0
	
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111, aspect='auto',projection='3d')
	plt.jet()

	# Plot contour surfaces
	_ = ax.contourf(
	X[:, :, -1], Y[:, :, -1], d3d[:, :, frames[0]].transpose(), #x,y,z
	zdir='z', offset=0, **kw
	)

	_ = ax.contourf(
	X[0, :, :], d3d[:, frames[2], :], Z[0, :, :],
	zdir='y', offset=Y.min(), **kw
	)
	
	C = ax.contourf(
	d3d[frames[1], :, :], Y[:, -1, :], Z[:, -1, :],
	zdir='x', offset=X.max(), **kw
	)

	plt.gca().set_xlabel("X",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z",fontsize='large', fontweight='normal')

	xmin, xmax = X.min(), X.max()
	ymin, ymax = Y.min(), Y.max()
	zmin, zmax = Z.min(), Z.max()
	ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
	plt.gca().invert_zaxis()

	# Colorbar
	if 'barlabel' in kw.keys():
		cbar=fig.colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format= "%.2f", label=kw['barlabel'])
		cbar.ax.locator_params(nbins=5)
		kwargs.__delitem__('barlabel')

	if figname is not None:
		if 'cmap' in kwargs.keys():
			kwargs.__delitem__('cmap')
		plt.savefig(figname,**kwargs)
	
	if showf:
		plt.show()
	else:
		if close:
			plt.close() #or plt.clear() ?
		
def framebox(x1,x2,y1,y2,c=None,lw=None):
	'''
	framebox: for drawing a frame box
	
	By Yangkang Chen
	June, 2022
	
	INPUT
	x1,x2,y1,y2: intuitive
	
	EXAMPLE I
	from pyseistr.plot import framebox
	from pyseistr.synthetics import gensyn
	from matplotlib import pyplot as plt
	d=gensyn();
	plt.imshow(d);
	framebox(200,400,200,300);
	plt.show()

	EXAMPLE II
	from pyseistr.plot import framebox
	from pyseistr.synthetics import gensyn
	from matplotlib import pyplot as plt
	d=gensyn();
	plt.imshow(d);
	framebox(200,400,200,300,c='g',lw=4);
	plt.show()
	
	'''
	
	if c is None:
		c='r';
	if lw is None:
		lw=2;

	plt.plot([x1,x2],[y1,y1],linestyle='-',color=c,linewidth=lw);
	plt.plot([x1,x2],[y2,y2],linestyle='-',color=c,linewidth=lw);
	plt.plot([x1,x1],[y1,y2],linestyle='-',color=c,linewidth=lw);
	plt.plot([x2,x2],[y1,y2],linestyle='-',color=c,linewidth=lw);

	
	return

def cube_visual(data, dt=0.004, dz=1, xlabel="Inline #", ylabel="Crossline #", zlabel="Time (s)", title=None, save_path=None):
    """
    Visualize a 3D seismic data cube.

    Parameters:
        data (numpy.ndarray): 3D array of seismic data (nz, nx, ny).
        dt (float): Sampling interval for the z-axis (e.g., time interval in seconds).
        dz (float): Sampling interval for the x and y axes (e.g., spatial interval).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
        title (str): Title of the plot.
        save_path (str): File path to save the plot. If None, the plot will not be saved.

    Returns:
        None
    """
    nz, nx, ny = data.shape
    z = np.arange(nz) * dt
    x = np.arange(nx) * dz
    y = np.arange(ny) * dz

    plt.figure(figsize=(6, 6))
    plot3d(data, z=z, x=x, y=y, showf=False, close=False)  # Assuming plot3d is defined elsewhere
    plt.gca().set_xlabel(xlabel, fontsize='large', fontweight='normal')
    plt.gca().set_ylabel(ylabel, fontsize='large', fontweight='normal')
    plt.gca().set_zlabel(zlabel, fontsize='large', fontweight='normal')
    if title:
        plt.title(title, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(fname=save_path, format='png', dpi=300)
    
    plt.show()

def patch3d(A,l1=4,l2=4,l3=4,s1=2,s2=2,s3=2,mode=1):
	"""
	patch3d: decompose 3D data into patches:
	
	INPUT
	D: input image
	mode: patching mode
	l1: first patch size
	l2: second patch size
	l3: third patch size
	s1: first shifting size
	s2: second shifting size
	s3: third shifting size
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	Oct, 2017
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	EXAMPLE
	sgk_denoise() in pyseisdl/denoise.py
	"""

	[n1,n2,n3]=A.shape;

	if mode==1: 	#possible for other patching options
	
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([s1-tmp,n2,n3])),axis=0);

		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],s2-tmp,n3])),axis=1);

		tmp=np.mod(n3-l3,s3);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],A.shape[1],s3-tmp])),axis=2);	#concatenate along the third dimension

		[N1,N2,N3]=A.shape;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					if i1==0 and i2==0 and i3==0:
						X=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],[l1*l2*l3,1],order='F');
					else:
						tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],[l1*l2*l3,1],order='F');
						X=np.concatenate((X,tmp),axis=1);
	else:
		#not written yet
		pass;
		
	return X

def patch3d_inv( X,n1,n2,n3,l1=4,l2=4,l3=4,s1=2,s2=2,s3=2,mode=1):
	"""
	patch3d_inv: insert patches into the 3D data
	
	INPUT
	D: input image
	mode: patching mode
	n1: first dimension size
	n1: second dimension size
	n3: third dimension size
	l1: first patch size
	l2: second patch size
	l3: third patch size
	s1: first shifting size
	s2: second shifting size
	s3: third shifting size
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	Oct, 2017
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	Marich, 31, 2020, 2D->3D
	
	EXAMPLE
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options
	
		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		tmp3=np.mod(n3-l3,s3);
		if tmp1!=0 and tmp2!=0 and tmp3!=0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3]);
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3]);

		if tmp1!=0 and tmp2!=0 and tmp3==0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3]);
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3]);
	
		if tmp1!=0 and tmp2==0 and tmp3==0:
			A=np.zeros([n1+s1-tmp1,n2,n3]);
			mask=np.zeros([n1+s1-tmp1,n2,n3]);
	
		if tmp1==0 and tmp2!=0 and tmp3==0:
			A=np.zeros([n1,n2+s2-tmp2,n3]);
			mask=np.zeros([n1,n2+s2-tmp2,n3]);
	
		if tmp1==0 and tmp2==0 and tmp3!=0:
			A=np.zeros([n1,n2,n3+s3-tmp3]);
			mask=np.zeros([n1,n2,n3+s3-tmp3]);
	
		if tmp1==0 and tmp2==0  and tmp3==0:
			A=np.zeros([n1,n2,n3]);
			mask=np.zeros([n1,n2,n3]);
	
		[N1,N2,N3]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					id=id+1;
					A[i1:i1+l1,i2:i2+l2,i3:i3+l3]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.reshape(X[:,id],[l1,l2,l3],order='F');
					mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.ones([l1,l2,l3]);
		A=A/mask;
	
		A=A[0:n1,0:n2,0:n3];
	else:
		#not written yet
		pass;
	return A


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range