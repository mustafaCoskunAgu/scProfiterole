import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
from prototype_loss import *
from sklearn.cluster import KMeans
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#import dgl.function as fn
import math
#import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
#from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv

'''
part of code is borrowed from https://github.com/CRIPAC-DIG/GRACE
'''


# mask feature function
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from utils import cheby  # make sure this is correctly implemented
from torch_geometric.utils import add_self_loops, get_laplacian
import math


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:49:54 2024

@author: Administrator
"""

import math
import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
import torch.nn.functional as F
from utils import cheby
from torch.nn import Parameter
from scipy.io import savemat



from typing import Optional, Tuple
import math
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from scipy.special import legendre

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Parameter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.linalg import expm
import scipy.io as sio

from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh
import random

from scipy.optimize import curve_fit

from scipy.special import gamma, factorial
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, add_self_loops

# -----------------------------------------------------------
#   Delete the above, my code starts from here
#------------------------------------------------------------


# Helper functions
import scipy.sparse as ss
import scipy.sparse.linalg as ssla
import numpy.random as nr
import matplotlib.pyplot as plt

def matrix_normalize(W,mode='s'):
	"""
	Normalize a weighted adjacency matrix.

	Args:
		W: weighted adjacency matrix
		mode: string indicating the style of normalization;
			's': Symmetric scaling by the degree (default)
			'r': Normalize to row-stochastic
			'c': Normalize to col-stochastic

	Output:
		N: a normalized adjacency matrix or stochastic matrix (in sparse form)
	"""

	dc = np.asarray(W.sum(0)).squeeze()
	dr = np.asarray(W.sum(1)).squeeze()
	[i,j,wij] = ss.find(W)

	# Normalize in desired style
	if mode in 'sl':
		wij = wij/np.sqrt(dr[i]*dc[j])
	elif mode == 'r':
		wij = wij/dr[i]
	elif mode == 'c':
		wij = wij/dc[j]
	else:
		raise ValueError('Unknown mode!')

	N = ss.csr_matrix((wij,(i,j)),shape=W.shape)
	return N


def moments_cheb_dos(A,n,nZ=100,N=10,kind=1):
	"""
	Compute a column vector of Chebyshev moments of the form c(k) = tr(T_k(A)) 
	for k = 0 to N-1. This routine does no scaling; the spectrum of A should 
	already lie in [-1,1]. The traces are computed via a stochastic estimator 
	with nZ probe

	Args:
		A: Matrix or function apply matrix (to multiple RHS)
		n: Dimension of the space
		nZ: Number of probe vectors with which we compute moments
		N: Number of moments to compute
		kind: 1 or 2 for first or second kind Chebyshev functions
		 	(default = 1)

	Output:
		c: a column vector of N moment estimates
		cs: standard deviation of the moment estimator 
			(std/sqrt(nZ))
	"""

	# Create a function handle if given a matrix 
	if callable(A):
		Afun = A
	else:
		if isinstance(A,np.ndarray):
			A = ss.csr_matrix(A)
		Afun = lambda x: A*x

	if N < 2:
		N = 2

	# Set up random probe vectors (allowed to be passed in)
	if not isinstance(nZ,int):
		Z = nZ
		nZ = Z.shape[1]
	else:
		Z = np.sign(nr.randn(n,nZ))

	# Estimate moments for each probe vector
	cZ = moments_cheb(Afun,Z,N,kind)
	c = np.mean(cZ,1)
	cs = np.std(cZ,1,ddof=1)/np.sqrt(nZ)

	c = c.reshape([N,-1])
	cs = cs.reshape([N,-1])
	return c,cs



def moments_cheb(A,V,N=10,kind=1):
	"""
	Compute a column vector of Chebyshev moments of the form c(k) = v'*T_k(A)*v 
	for k = 0 to N-1. This routine does no scaling; the spectrum of A should 
	already lie in [-1,1]

	Args:
		A: Matrix or function apply matrix (to multiple RHS)
		V: Starting vectors
		N: Number of moments to compute
		kind: 1 or 2 for first or second kind Chebyshev functions
			(default = 1)

	Output:
		c: a length N vector of moments
	"""

	if N<2:
		N = 2

	if not isinstance(V,np.ndarray):
		V = V.toarray()

	# Create a function handle if given a matrix
	if  callable(A):
		Afun = A
	else:
		if isinstance(A,np.ndarray):
			A = ss.csr_matrix(A)
		Afun = lambda x: A*x

	n,p = V.shape
	c = np.zeros((N,p))

	# Run three-term recurrence to compute moments
	TVp = V
	TVk = kind*Afun(V)
	c[0] = np.sum(V*TVp,0)
	c[1] = np.sum(V*TVk,0)
	for i in range(2,N):
		TV = 2*Afun(TVk) - TVp
		TVp = TVk
		TVk = TV
		c[i] = sum(V*TVk,0)

	return c

def moments_cheb_ldos(A,n,nZ=100,N=10,kind=1):
	"""
	Compute a column vector of Chebyshev moments of the form 
	c(k,j) = [T_k(A)]_jj for k = 0 to N-1. This routine does no scaling; the 
	spectrum of A should already lie in [-1,1]. The diagonal entries are 
	computed by a stochastic estimator

	Args:
		A: Matrix or function apply matrix (to multiple RHS)
		n: Dimension of the space
		nZ: Number of probe vectors with which we compute moments
		N: Number of moments to compute
		kind: 1 or 2 for first or second kind Chebyshev functions
		 	(default = 1)

	Output:
		c: a (N,n) matrix of moments
		cs: standard deviation of the moment estimator 
			(std/sqrt(nZ))
	"""
	
	# Create a function handle if given a matrix
	if callable(A):
		Afun = A
	else:
		if isinstance(A,np.ndarray):
			A = ss.csr_matrix(A)
		Afun = lambda x: A*x

	if N < 2:
		N = 2

	# Set up random probe vectors (allowed to be passed in)
	if not isinstance(nZ,int):
		Z = nZ
		nZ = Z.shape[1]
	else:
		Z = np.sign(nr.randn(n,nZ))

	# Run three-term recurrence to estimate moments.
	# Use the stochastic diagonal estimator of Bekas and Saad
	# http://www-users.cs.umn.edu/~saad/PDF/usmi-2005-082.pdf

	c = np.zeros((N,n))
	cs = np.zeros((N,n))

	TZp = Z
	X = Z*TZp
	c[0] = np.mean(X,1).T
	cs[0] = np.std(X,1,ddof=1).T

	TZk = kind*Afun(Z)
	X = Z*TZk
	c[1] = np.mean(X,1).T
	cs[1] = np.std(X,1,ddof=1).T

	for i in range(2,N):
		TZ = 2*Afun(TZk) - TZp
		TZp = TZk
		TZk = TZ
		X = Z*TZk
		c[i] = np.mean(X,1).T
		cs[i] = np.std(X,1,ddof=1).T

	cs = cs/np.sqrt(nZ)

	c = c.reshape([N,-1])
	cs = cs.reshape([N,-1])
	return c,cs

def plot_cheb_argparse(npts,c,xx0=-1,ab=np.array([1,0])):
	"""
	Handle argument parsing for plotting routines. Should not be called directly
	by users.

	Args:
		npts: Number of points in a default mesh
		c: Vector of moments
		xx0: Input sampling mesh (original coordinates)
		ab: Scaling map parameters

	Output:
		c: Vector of moments
		xx: Input sampling mesh ([-1,1] coordinates)
		xx0: Input sampling mesh (original coordinates)
		ab: Scaling map parameters
	"""

	if isinstance(xx0,int):
		# only c is given
		xx0 = np.linspace(-1+1e-8,1-1e-8,npts)
		xx = xx0
	else:
		if len(xx0)==2:
			# parameters are c, ab
			ab = xx0
			xx = np.linspace(-1+1e-8,1-1e-8,npts)
			xx0 = ab[0]*xx+ab[1]
		else:
			# parameteres are c, xx0
			xx=xx0

	# All parameters specified
	if not (ab==[1,0]).all():
		xx = (xx0-ab[1])/ab[0]

	return c,xx,xx0,ab

def plot_cheb(varargin,pflag=True):
	"""
	Given a set of first-kind Chebyshev moments, compute the associated density.
	Output a plot of the density function by default

	Args:
		c: Vector of Chebyshev moments (on [-1,1])
		xx: Evaluation points (defaults to mesh of 1001 pts)
		ab: Mapping parameters (default to identity)
		pflag: Option to output the plot

	Output:
		yy: Density evaluated at xx mesh
	"""

	# Parse arguments
	c,xx,xx0,ab = plot_cheb_argparse(1001,*varargin)

	# Run the recurrence
	kind = 1
	N = len(c)
	P0 = xx*0+1
	P1 = kind*xx
	yy =c[0]/(3-kind)+c[1]*xx

	for idx in np.arange(2,N):
		Pn = 2*(xx*P1)-P0
		yy += c[idx]*Pn
		P0 = P1
		P1 = Pn

	# Normalization
	if kind == 1:
		yy = (2/np.pi/ab[0])*(yy/(1e-12+np.sqrt(1-xx**2)))
	else:
		yy = (2/np.pi/ab[0])*(yy*np.sqrt(1-xx**2))

	# Plot by default
	if pflag:
		plt.plot(xx0,yy)
		plt.ion()
		plt.show()
		plt.pause(1)
		plt.clf()
		# input('Press [enter] to continue.')

	yy.reshape([1,-1])
	return yy

def plot_cheb_ldos(varargin,pflag=True):
	"""
	Given a set of first-kind Chebyshev moments, compute the associated local 
	density. Output a plot of the local density functions by default.

	Args:
		c: Vector of Chebyshev moments (on [-1,1])
		xx: Evaluation points (defaults to mesh of 1001 pts)
		ab: Mapping parameters (default to identity)
		pflag: Option to output the plot

	Output:
		yy: Density evaluated at xx mesh (size nnodes-by-nmesh)
		index: Index for spectral re-ordering
	"""

	# Parse arguments
	c,xx,xx0,ab = plot_cheb_argparse(51,*varargin)

	# Run the recurrence to compute CDF
	nmoment,nnodes = c.shape
	txx = np.arccos(xx)
	yy = c[0].reshape((nnodes,1))*(txx-np.pi)/2
	for idx in np.arange(1,nmoment):
		yy = yy +c[idx].reshape((nnodes,1))*np.sin(idx*txx)/idx

	# Difference the CDF to compute histogram
	yy *= -2/np.pi
	yy = yy[:,1:]-yy[:,0:-1]

	# Compute the sorted histogram
	(u,s,v) = ssla.svds(yy,1)
	index = np.argsort(np.squeeze(u))

	# Plot by default
	if pflag:
		fig,ax = plt.subplots()
		yr = np.array([1,nnodes])
		xr = np.array([xx0[0]+xx0[1],xx0[-1]+xx0[-2]],dtype='float')/2
		im = ax.imshow(yy[index,:],extent=np.append(xr,yr),aspect=1.5/nnodes)
		fig.colorbar(im)
		plt.ion()
		plt.show()
		plt.pause(1)
		plt.clf()

	index = np.float64(index.reshape([-1,1]))
	return yy,index

def plot_chebhist(varargin,pflag=False):
	"""
	Given a (filtered) set of first-kind Chebyshev moments, compute the integral
	of the density:
		int_0^s (2/pi)*sqrt(1-x^2)*( c(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )
	Output a histogram of cumulative density function by default.

	Args:
		c: Vector of Chebyshev moments (on [-1,1])
		xx: Evaluation points (defaults to mesh of 21 pts)
		ab: Mapping parameters (default to identity)
		pflag: Option to output the plot

	Output:
		yy: Estimated counts on buckets between xx points
	"""

	# Parse arguments
	c,xx,xx0,ab = plot_cheb_argparse(101,*varargin)

	# Compute CDF and bin the difference
	yy = plot_chebint((c,xx0,ab),pflag=False)
	yy = yy[1:]-yy[:-1]
	xm = (xx0[1:]+xx0[:-1])/2
	
	# Plot by default
	if pflag:
		plt.bar(xm,yy,align='center',width=0.1)
		plt.ion()
		plt.show()
		plt.pause(1)
		plt.clf()

	return xm,yy

def plot_chebint(varargin,pflag=True):
	"""
	Given a (filtered) set of first-kind Chebyshev moments, compute the integral
	of the density:
		int_0^s (2/pi)*sqrt(1-x^2)*( c(0)/2+sum_{n=1}^{N-1}c_nT_n(x) )
	Output a plot of cumulative density function by default.

	Args:
		c: Array of Chebyshev moments (on [-1,1])
		xx: Evaluation points (defaults to mesh of 1001 pts)
		ab: Mapping parameters (default to identity)
		pflag: Option to output the plot

	Output:
		yy: Estimated cumulative density up to each xx point
	"""

	# Parse arguments
	c,xx,xx0,ab = plot_cheb_argparse(1001,*varargin)

	N = len(c)
	txx = np.arccos(xx)
	yy = c[0]*(txx-np.pi)/2
	for idx in np.arange(1,N):
		yy += c[idx]*np.sin(idx*txx)/idx

	yy *= -2/np.pi

	# Plot by default
	if pflag:
		plt.plot(xx0,yy)
		plt.ion()
		plt.show()
		plt.pause(1)
		plt.clf()

	return yy

def plot_chebp(varargin,pflag=True):
	"""
	Given a set of first-kind Chebyshev moments, compute the associated 
	polynomial (*NOT* a density). Output a plot of the polynomial by default.

	Args:
		c: Vector of Chebyshev moments (on [-1,1])
		xx: Evaluation points (defaults to mesh of 1001 pts)
		ab: Mapping parameters (default to identity)
		pflag: Option to output the plot

	Output:
		yy: Polynomial evaluated at xx mesh
	"""

	# Parse arguments
	c,xx,xx0,ab = plot_cheb_argparse(1001,*varargin)
    
	# Run the recurrence
	kind = 1
	N = len(c)
	P0 = xx*0+1
	P1 = kind*xx
	yy = c[0]+c[1]*xx
	for idx in np.arange(2,N):
		Pn = 2*(xx*P1)-P0
		yy += c[idx]*Pn
		P0 = P1
		P1 = Pn

	# Plot by default
	if pflag:
		plt.plot(xx0,yy)
		plt.ion()
		plt.show()
		plt.pause(1)
		plt.clf()

	return yy



import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
def _hkt(eivals, timescales):
    """
    Computes heat kernel trace from given eigenvalues, timescales, and normalization.

    For precise definition, please refer to "NetLSD: Hearing the Shape of a Graph" by A. Tsitsulin, D. Mottin, P. Karras, A. Bronstein, E. Müller. Published at KDD'18.
    
    Parameters
    ----------
    eivals : numpy.ndarray
        Eigenvalue vector
    timescales : numpy.ndarray
        Vector of discrete timesteps for the kernel computation
    normalization : str or numpy.ndarray
        Either 'empty', 'complete' or None.
        If None or any ther value, return unnormalized heat kernel trace.
        For the details how 'empty' and 'complete' are computed, please refer to the paper.
        If np.ndarray, they are treated as exact normalization constants
    normalized_laplacian: bool
        Defines whether the eigenvalues came from the normalized Laplacian. It only affects 'complete' normalization.

    Returns
    -------
    numpy.ndarray
        Heat kernel trace signature

    """
    nv = eivals.shape[0]
    hkt = np.zeros(timescales.shape)
    for idx, t in enumerate(timescales):
        hkt[idx] = np.sum(np.exp(-t * eivals))

    return hkt / nv


def mat_to_laplacian(mat, normalized):
    """
    Converts a sparse or dence adjacency matrix to Laplacian.
    
    Parameters
    ----------
    mat : obj
        Input adjacency matrix. If it is a Laplacian matrix already, return it.
    normalized : bool
        Whether to use normalized Laplacian.
        Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.

    Returns
    -------
    obj
        Laplacian of the input adjacency matrix

    Examples
    --------
    >>> mat_to_laplacian(numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), False)
    [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]

    """
    if sps.issparse(mat):
        if np.all(mat.diagonal()>=0): # Check diagonal
            if np.all((mat-sps.diags(mat.diagonal())).data <= 0): # Check off-diagonal elements
                return mat
    else:
        if np.all(np.diag(mat)>=0): # Check diagonal
            if np.all(mat - np.diag(mat) <= 0): # Check off-diagonal elements
                return mat
    deg = np.squeeze(np.asarray(mat.sum(axis=1)))
    if sps.issparse(mat):
        L = sps.diags(deg) - mat
    else:
        L = np.diag(deg) - mat
    if not normalized:
        return L
    with np.errstate(divide='ignore'):
        sqrt_deg = 1.0 / np.sqrt(deg)
    sqrt_deg[sqrt_deg==np.inf] = 0
    if sps.issparse(mat):
        sqrt_deg_mat = sps.diags(sqrt_deg)
    else:
        sqrt_deg_mat = np.diag(sqrt_deg)
    return sqrt_deg_mat.dot(L).dot(sqrt_deg_mat)


def updown_linear_approx(eigvals_lower, eigvals_upper, nv):
    """
    Approximates Laplacian spectrum using upper and lower parts of the eigenspectrum.
    
    Parameters
    ----------
    eigvals_lower : numpy.ndarray
        Lower part of the spectrum, sorted
    eigvals_upper : numpy.ndarray
        Upper part of the spectrum, sorted
    nv : int
        Total number of nodes (eigenvalues) in the graph.

    Returns
    -------
    numpy.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> updown_linear_approx([1, 2, 3], [7, 8, 9], 9)
    array([1,  2,  3,  4,  5,  6,  7,  8,  9])

    """
    nal = len(eigvals_lower)
    nau = len(eigvals_upper)
    if nv < nal + nau:
        raise ValueError('Number of supplied eigenvalues ({0} lower and {1} upper) is higher than number of nodes ({2})!'.format(nal, nau, nv))
    ret = np.zeros(nv)
    ret[:nal] = eigvals_lower
    ret[-nau:] = eigvals_upper
    ret[nal-1:-nau+1] = np.linspace(eigvals_lower[-1], eigvals_upper[0], nv-nal-nau+2)
    return ret


def eigenvalues_auto(mat, n_eivals=100):
    """
    Automatically computes the spectrum of a given Laplacian matrix.
    
    Parameters
    ----------
    mat : numpy.ndarray or scipy.sparse
        Laplacian matrix
    n_eivals : string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.

    Returns
    -------
    np.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> eigenvalues_auto(numpy.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]), 'auto')
    array([0, 3, 3])

    """
    do_full = True
    n_lower = 100
    n_upper = 100
    nv = mat.shape[0]
    if n_eivals == 'auto':
        if mat.shape[0] > 1024:
            do_full = False
    if n_eivals == 'full':
        do_full = True
    if isinstance(n_eivals, int):
        n_lower = n_upper = n_eivals
        do_full = False
    if isinstance(n_eivals, tuple):
        n_lower, n_upper = n_eivals
        do_full = False
    if do_full and sps.issparse(mat):
        mat = mat.todense()
    if sps.issparse(mat):
        if n_lower == n_upper:
            tr_eivals = spsl.eigsh(mat, 2*n_lower, which='BE', return_eigenvectors=False)
            return updown_linear_approx(tr_eivals[:n_upper], tr_eivals[n_upper:], nv)
        else:
            lo_eivals = spsl.eigsh(mat, n_lower, which='SM', return_eigenvectors=False)[::-1]
            up_eivals = spsl.eigsh(mat, n_upper, which='LM', return_eigenvectors=False)
            return updown_linear_approx(lo_eivals, up_eivals, nv)
    else:
        if do_full:
            return spl.eigvalsh(mat)
        else:
            lo_eivals = spl.eigvalsh(mat, eigvals=(0, n_lower-1))
            up_eivals = spl.eigvalsh(mat, eigvals=(nv-n_upper-1, nv-1))
            return updown_linear_approx(lo_eivals, up_eivals, nv)


def eigenvalues_bound(mat, n_eivals=100):
    """
    Automatically computes the spectrum of a given Laplacian matrix.
    
    Parameters
    ----------
    mat : numpy.ndarray or scipy.sparse
        Laplacian matrix
    n_eivals : string or int or tuple
        Number of eigenvalues to compute / use for approximation.
        If string, we expect either 'full' or 'auto', otherwise error will be raised. 'auto' lets the program decide based on the faithful usage. 'full' computes all eigenvalues.
        If int, compute n_eivals eigenvalues from each side and approximate using linear growth approximation.
        If tuple, we expect two ints, first for lower part of approximation, and second for the upper part.

    Returns
    -------
    np.ndarray
        Vector of approximated eigenvalues

    Examples
    --------
    >>> eigenvalues_auto(numpy.array([[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]), 'auto')
    array([0, 3, 3])

    """
    do_full = True
    n_lower = 100
    n_upper = 100
    nv = mat.shape[0]
    if n_eivals == 'auto':
        if mat.shape[0] > 1024:
            do_full = False
    if n_eivals == 'full':
        do_full = True
    if isinstance(n_eivals, int):
        n_lower = n_upper = n_eivals
        do_full = False
    if isinstance(n_eivals, tuple):
        n_lower, n_upper = n_eivals
        do_full = False
    if do_full and sps.issparse(mat):
        mat = mat.todense()
    if sps.issparse(mat):
        if n_lower == n_upper:
            tr_eivals = spsl.eigsh(mat, 2*n_lower, which='BE', return_eigenvectors=False)
            return np.concatenate((tr_eivals[:n_upper], tr_eivals[n_upper:]), axis=None) 
        else:
            lo_eivals = spsl.eigsh(mat, n_lower, which='SM', return_eigenvectors=False)[::-1]
            up_eivals = spsl.eigsh(mat, n_upper, which='LM', return_eigenvectors=False)
            return np.concatenate((lo_eivals, up_eivals), axis=None)
    else:
        if do_full:
            return spl.eigvalsh(mat)
        else:
            lo_eivals = spl.eigvalsh(mat, eigvals=(0, n_lower-1))
            up_eivals = spl.eigvalsh(mat, eigvals=(nv-n_upper-1, nv-1))
            return np.concatenate((lo_eivals, up_eivals), axis=None)

def func(x, p1,p2):
  return p1*np.cos(p2*x) + p2*np.sin(p1*x)

import scipy.linalg
from sklearn.neighbors import KernelDensity
from scipy.sparse.linalg import eigsh

    
    

#! /usr/bin/env python
#
def imtqlx ( n, d, e, z ):

#*****************************************************************************80
#
## IMTQLX diagonalizes a symmetric tridiagonal matrix.
#
#  Discussion:
#
#    This routine is a slightly modified version of the EISPACK routine to
#    perform the implicit QL algorithm on a symmetric tridiagonal matrix.
#
#    The authors thank the authors of EISPACK for permission to use this
#    routine.
#
#    It has been modified to produce the product Q' * Z, where Z is an input
#    vector and Q is the orthogonal matrix diagonalizing the input matrix.
#    The changes consist (essentially) of applying the orthogonal 
#    transformations directly to Z as they are generated.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    15 June 2015
#
#  Author:
#
#    John Burkardt.
#
#  Reference:
#
#    Sylvan Elhay, Jaroslav Kautsky,
#    Algorithm 655: IQPACK, FORTRAN Subroutines for the Weights of
#    Interpolatory Quadrature,
#    ACM Transactions on Mathematical Software,
#    Volume 13, Number 4, December 1987, pages 399-415.
#
#    Roger Martin, James Wilkinson,
#    The Implicit QL Algorithm,
#    Numerische Mathematik,
#    Volume 12, Number 5, December 1968, pages 377-383.
#
#  Parameters:
#
#    Input, integer N, the order of the matrix.
#
#    Input, real D(N), the diagonal entries of the matrix.
#
#    Input, real E(N), the subdiagonal entries of the
#    matrix, in entries E(1) through E(N-1). 
#
#    Input, real Z(N), a vector to be operated on.
#
#    Output, real LAM(N), the diagonal entries of the diagonalized matrix.
#
#    Output, real QTZ(N), the value of Q' * Z, where Q is the matrix that 
#    diagonalizes the input symmetric tridiagonal matrix.
#
  import numpy as np
  #from r8_epsilon import r8_epsilon
  
  from sys import exit

  lam = np.zeros ( n )
  for i in range ( 0, n ):
    lam[i] = d[i]

  qtz = np.zeros ( n )
  for i in range ( 0, n ):
    qtz[i] = z[i]

  if ( n == 1 ):
    return lam, qtz

  itn = 30

  prec = 2.220446049250313E-016

  e[n-1] = 0.0

  for l in range ( 1, n + 1 ):

    j = 0

    while ( True ):

      for m in range ( l, n + 1 ):

        if ( m == n ):
          break

        if ( abs ( e[m-1] ) <= prec * ( abs ( lam[m-1] ) + abs ( lam[m] ) ) ):
          break

      p = lam[l-1]

      if ( m == l ):
        break

      if ( itn <= j ):
        print ( '' )
        print ( 'IMTQLX - Fatal error!' )
        print ( '  Iteration limit exceeded.' )
        exit ( 'IMTQLX - Fatal error!' )

      j = j + 1
      g = ( lam[l] - p ) / ( 2.0 * e[l-1] )
      r = np.sqrt ( g * g + 1.0 )

      if ( g < 0.0 ):
        t = g - r
      else:
        t = g + r

      g = lam[m-1] - p + e[l-1] / ( g + t )
 
      s = 1.0
      c = 1.0
      p = 0.0
      mml = m - l

      for ii in range ( 1, mml + 1 ):

        i = m - ii
        f = s * e[i-1]
        b = c * e[i-1]

        if ( abs ( g ) <= abs ( f ) ):
          c = g / f
          r = np.sqrt ( c * c + 1.0 )
          e[i] = f * r
          s = 1.0 / r
          c = c * s
        else:
          s = f / g
          r = np.sqrt ( s * s + 1.0 )
          e[i] = g * r
          c = 1.0 / r
          s = s * c

        g = lam[i] - p
        r = ( lam[i-1] - g ) * s + 2.0 * c * b
        p = s * r
        lam[i] = g + p
        g = c * r - b
        f = qtz[i]
        qtz[i]   = s * qtz[i-1] + c * f
        qtz[i-1] = c * qtz[i-1] - s * f

      lam[l-1] = lam[l-1] - p
      e[l-1] = g
      e[m-1] = 0.0

  for ii in range ( 2, n + 1 ):

     i = ii - 1
     k = i
     p = lam[i-1]

     for j in range ( ii, n + 1 ):

       if ( lam[j-1] < p ):
         k = j
         p = lam[j-1]

     if ( k != i ):

       lam[k-1] = lam[i-1]
       lam[i-1] = p

       p        = qtz[i-1]
       qtz[i-1] = qtz[k-1]
       qtz[k-1] = p

  return lam, qtz


#! /usr/bin/env python
#
def p_polynomial_zeros ( nt ):

#*****************************************************************************80
#
## P_POLYNOMIAL_ZEROS: zeros of Legendre function P(n,x).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    16 March 2016
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    Input, integer NT, the order of the rule.
#
#    Output, real T(NT), the zeros.
#


  a = np.zeros ( nt )

  b = np.zeros ( nt )

  for i in range ( 0, nt ):
    ip1 = i + 1
    b[i] = ip1 / np.sqrt ( 4 * ip1 * ip1 - 1 )

  c = np.zeros ( nt )
  c[0] = np.sqrt ( 2.0 )

  t, w = imtqlx ( nt, a, b, c )

  return  t+1# 0.9*t#for [0, 2] interval


def j_polynomial_zeros ( nt, alpha, beta):

#*****************************************************************************80
#
## P_POLYNOMIAL_ZEROS: zeros of Legendre function P(n,x).
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    19 October 2023
#
#  Author:
#
#    Dr. Mustafa Coşkun
#
#  Parameters:
#
#    Input, integer NT, the order of the rule, upper and lower are bounds.
#
#    Output, real T(NT), the zeros.
#

  ab = alpha + beta
  abi = 2.0 + ab
  # define the zero-th moment
  zemu = (np.power(2.0,(ab + 1.0))*gamma(alpha + 1.0)*gamma(beta + 1.0))/gamma(abi)
  
  
  x = np.zeros(nt)
  bj = np.zeros(nt)
  
  
  x[0] = (beta -alpha)/abi
  bj[0] = np.sqrt(4.0*(1.0 + alpha)*(1.0 + beta)/((abi + 1.0)*abi*abi))
  a2b2 = beta*beta-alpha*alpha

  for i in range ( 2, nt +1 ):
    abi = 2.0*i + ab
    x[i-1] = a2b2 / ((abi -2.0)*abi)
    abi = np.power(abi,2)
    bj[i-1] = np.sqrt((4.0*i*(i+alpha)*(i + beta)* (i + ab))/((abi -1.0)*abi))

 #bjs = np.sqrt(bj)
  c = np.zeros ( nt )
  c[0] = np.sqrt ( zemu )

  t, w = imtqlx ( nt, x, bj, c )

  return t+1 # for 0.9*t#[0, 2] interval

def g_fullRWR(x):
    return (1)/(1 - x) 
    #return x/(1-x) - x**2 


def g_0(x, alpha  = 0.9):
    return (alpha)/(1 - x) 
def g_1(x):
    return (1)/(1 - x)
def g_2(x):
    return ((x)/(1 - x))
def g_3(x):
    return (x**2)/(1 - x)   
def g_4(x):
    return (1)/(1 + 25*x**2)

def g_implicit_cora(x):
  p1 = 1.5820425413644672 
  p2 =  0.389804848348408
  return p1*np.cos(p2*x) + p2*np.sin(p1*x)
def g_implicit_citeseer(x):
  p1 = 3.9219500173434954 
  p2 =  -0.17644030889351756
  print("Is implicit citeseer function called")
  return p1*np.cos(p2*x) + p2*np.sin(p1*x) 

def f0_betaL(x):
    """
    Compute the Beta kernel polynomial f_0(x) for a given degree K.
    f_0(x) = (K+1) * (1 - x/2)^K
    """
    K = 6
    return (K + 1) * (1 - x / 2) ** K


def f0_beta(x, K = 6):
    """
    Compute the Beta kernel polynomial f_0(x) for a given degree K.
    f_0(x) = (K+1) * (1 - x/2)^K
    """
    
    return (K + 1) * ((1 + x) / 2) ** K

def g_par(x):
    return 1/(1 + x)

def g_appRWR(x,Ksteps):
    sum = 0
    for k in range(Ksteps):
        sum = sum + x**k
    return 0.1*sum
def g_hub_promoting(x):
    return 1/(1+3*x)**2
def g_low_heat(x,t = 2.0):
    return np.exp(-t + t*x)
def g_high_heat(x):
    return 1- np.exp(-2*x)
def g_heat(x,Ksteps):
    t = 5
    sum = 0
    for k in range(Ksteps):
        sum = sum + (t**k)/np.math.factorial(k)
    return np.math.exp(-sum)

def g_diffusionKernel(x):
    beta = 5.0
    alpha = beta/(1+beta)
    return (1-alpha)/(1-alpha*x)
    #return 1/(1 + beta - beta*x)
    # sum = 0
    # for k in range(20):
    #     sum = sum + ((beta/1 + beta )**k) * x*k
    # return (1/ 1 + beta)*sum
def g_band_rejection(x):
    return (1-np.exp(-3*(x-1)**2))

def g_band_pass(x):
    return np.exp(-2*(x-1)**2)

def g_low_pass(x):
    return np.exp(-5*x**2)
def g_high_pass(x):
    return 1-np.exp(-0.1*x)

def g_mix_high_low(x):
    return (1/(1-x))*(1-np.exp(-5*x**2))

def g_comb(x):
    return np.abs(np.sin(np.pi*x))
def g_my_heat(x):
    return np.exp(-3*(x-1)/x)
    
def filter_jackson(c):
    N = len(c)
    n = np.arange(N)
    tau = np.pi/(N+1)
    g = ((N-n+1)*np.cos(tau*n) + np.sin(tau*n)/np.tan(tau))/(N+1)
    c = np.multiply(g,c)
    return c

# def filter_jackson(c):
# 	"""
# 	Apply the Jackson filter to a sequence of Chebyshev	moments. The moments 
# 	should be arranged column by column.

# 	Args:
# 		c: Unfiltered Chebyshev moments

# 	Output:
# 		cf: Jackson filtered Chebyshev moments
# 	"""

# 	N = len(c)
# 	n = np.arange(N)
# 	tau = np.pi/(N+1)
# 	g = ((N-n+1)*np.cos(tau*n)+np.sin(tau*n)/np.tan(tau))/(N+1)
# 	g.shape = (N,1)
# 	c = g*c 
#     #print(c)
    
# 	return c

def g_Ours(x):
    sum = 1*1 + 1*x + 4*x**2 + 5*x**3
    return sum
def runge(x):
  """In some places the x range is expanded and the formula give as 1/(1+x^2)
  """
  return 1 / (1 +  x ** 2)

def polyfitA(x,y,n):

    m = x.size
    Q = np.ones((m, 1), dtype=object)
    H = np.zeros((n+1, n), dtype=object)
    k = 0
    j = 0
    for k in range(n):
        q = np.multiply(x,Q[:,k])
        #print(q)
        for j in range(k):
            H[j,k] = np.dot(Q[:,j].T,(q/m))
            q = q - np.dot(H[j,k],(Q[:,j]))
        H[k+1,k] = np.linalg.norm(q)/np.sqrt(m)
        Q = np.column_stack((Q, q/H[k+1,k]))
    #print(Q)
    #print(Q.shape)
    d = np.linalg.solve(Q.astype(np.float64), y.astype(np.float64))
    return d, H

def polyvalA(d,H,s):
    inputtype = H.dtype.type
    M = len(s)
    W = np.ones((M,1), dtype=inputtype)
    n = H.shape[1]
    #print("Complete H", H)
    k = 0
    j = 0
    for k in range(n):
        w = np.multiply(s,W[:,k])
        for j in range(k):
            #print( "H[j,k]",H[j,k])
            w = w -np.dot(H[j,k],( W[:,j]))
        W = np.column_stack((W, w/H[k+1,k]))
    y = W @ d
    return y, W


def t_polynomial_zeros(x0, x1, n):
  return (x1 - x0) * (np.cos((2*np.arange(1, n + 1) - 1)/(2*n)*np.pi) + 1) / 2  + x0

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def s_polynomial_zeros(n):
    temp = Parameter(torch.Tensor(n+1))
    temp.data.fill_(1.0)
    coe_tmp=F.relu(temp)
    coe=coe_tmp.clone()
    for i in range(n):
        coe[i]=coe_tmp[0]*cheby(i,math.cos((n+0.5)*math.pi/(n+1)))
        for j in range(1,n+1):
            x_j=math.cos((n-j+0.5)*math.pi/(n+1))
            coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
        coe[i]=2*coe[i]/(n+1)
    return coe

def poylfitA_Cheby(x,y,n,a,b):
    omega = (b-a)/2
    rho = -((b+a)/(b-a))
    
    IN = np.identity(n+1)
    X = np.diag(x)
    
    firstElement = (2/omega)*X + 2*rho*IN
    firstRow = np.concatenate((firstElement, -IN), axis=1)
    secondRow = np.concatenate((IN, np.zeros((n+1, n+1), dtype=object)), axis=1)
    Xcurly = np.concatenate((firstRow, secondRow), axis=0)
    
   
    m = x.size
    
    T = np.ones((m, 1), dtype=object)

    #This is just convert (n,) to (n,1) shape
    x = x.reshape(-1,1)
    #y = y.reshape(-1,1)
    Q = np.concatenate((x,T),axis = 0)

  
    H = np.zeros((n+1, n), dtype=object)
    k = 0
    j = 0
    for k in range(n):
        q = np.matmul(Xcurly,Q[:,k])
        #print(q)
        for j in range(k):
            H[j,k] = np.dot(Q[:,j].T,(q/m))
            q = q - np.dot(H[j,k],(Q[:,j]))
        H[k+1,k] = np.linalg.norm(q)/np.sqrt(m)
        Q = np.column_stack((Q, q/H[k+1,k]))
  
    newQ = Q[n+1:2*(n+1),:];

    d = np.linalg.solve(newQ.astype(np.float64), y.astype(np.float64))
    return d, H

def compare_fitA(f, x, Vander, Threeterm,x0, x1):
  y = f(x)
  n = x.size-1

  if(Vander):
      coefficients = Vandermonde(x, y)

  else:
      if(Threeterm):
          coefficients, H = poylfitA_Cheby(x,y,n,x0,x1)    
      else:
          coefficients, H = polyfitA(x, y, n)
  #K = coefficients.shape[0]
  # for k in range(K-1, -1, -1):
  #     print(coefficients[k], k)
  #print(coefficients)
  return coefficients

def m_polynomial_zeros (x0, x1, n):
    return  np.linspace(x0, x1,n)
def compare_fit_panelA(f, sampling, Vandermonde, Threeterm,degree, x0, x1,zoom=False):
   # Male equedistance
   #x = np.linspace(x0, x1,10)
   
   if (sampling == 'Monomial'):
       x = m_polynomial_zeros(x0, x1, degree)
   elif (sampling == 'Chebyshev'):    
       x = t_polynomial_zeros(x0, x1, degree)
   elif (sampling == 'Legendre'):
       x = p_polynomial_zeros(degree)
   elif (sampling == 'Jacobi'):    
       x = j_polynomial_zeros(degree,0,1)
   else:
    print ('Give proper polynomial to interpolate\n')
    print ('Calling Monimal as default\n')
    x = m_polynomial_zeros(x0, x1, degree)
    
   return compare_fitA(f, x, Vandermonde,Threeterm, x0,x1)


def compare_fit_panelAImplicit(y, Vandermonde, Threeterm,degree, x0, x1,zoom=False):
   # # Male equedistance
   # #x = np.linspace(x0, x1,10)
   # if (sampling == 'Monomial'):
   #     x = m_polynomial_zeros(x0, x1, degree)
   # elif (sampling == 'Chebyshev'):    
   #     x = t_polynomial_zeros(x0, x1, degree)
   # elif (sampling == 'Legendre'):
   #     x = p_polynomial_zeros(degree)
   # elif (sampling == 'Jacobi'):    
   #     x = j_polynomial_zeros(degree,0,1)
   # else:
   #  print ('Give proper polynomial to interpolate\n')
   #  print ('Calling Monimal as default\n')
   #  x = m_polynomial_zeros(x0, x1, degree)
   
   x = m_polynomial_zeros(-1, 1, degree)
    
   return compare_fitAImplicit(y,x, Vandermonde,Threeterm, -1,1)

def compare_fitAImplicit(y,x, Vander, Threeterm,x0, x1):
  # y = f(x)
  n = x.size-1

  if(Vander):
      coefficients = Vandermonde(x, y)

  else:
      if(Threeterm):
          coefficients, H = poylfitA_Cheby(x,y,n,x0,x1)    
      else:
          coefficients, H = polyfitA(x, y, n)
  #K = coefficients.shape[0]
  # for k in range(K-1, -1, -1):
  #     print(coefficients[k], k)
  #print(coefficients)
  return coefficients
def Vandermonde(x, y):
  """Return a polynomial fit of order n+1 to n points"""
  #z = np.polyfit(x, y, x.size + 1)
 
  V = np.vander(x) # Vandermonde matrix
  coeffs = np.linalg.solve(V, y) # f_nodes must be a column vector
  return coeffs


def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def take_rest(x, y):
    x.sort()
    y.sort()
    res = []
    j, jmax = 0, len(y)
    for i in range(0, len(x)):
        flag = False
        while j < jmax and y[j] <= x[i]:
            if y[j] == x[i]:
                flag = True
            j += 1
        if not flag:
            res.append(x[i])
    return res


def presum_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx-1] + h[idx-1]
    return temp

def preminus_tensor(h, initial_val):
    length = len(h) + 1
    temp = torch.zeros(length)
    temp[0] = initial_val
    for idx in range(1, length):
        temp[idx] = temp[idx-1] - h[idx-1]
    return temp

def reverse_tensor(h):
    temp = torch.zeros_like(h)
    length = len(temp)
    for idx in range(0, length):
        temp[idx] = h[length-1-idx]
    return temp


class ChebnetII_prop(MessagePassing):
    def __init__(self, K, Init=False, bias=True, **kwargs):
        super(ChebnetII_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K + 1))
        self.Init = Init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)
        if self.Init:
            for j in range(self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                self.temp.data[j] = x_j ** 2

    def forward(self, x, edge_index, edge_weight=None):
        coe_tmp = F.relu(self.temp)
        coe = coe_tmp.clone()

        for i in range(self.K + 1):
            coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
            for j in range(1, self.K + 1):
                x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
                coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
            coe[i] = 2 * coe[i] / (self.K + 1)

        # Compute Laplacian
        edge_index1, norm1 = get_laplacian(
            edge_index, edge_weight,
            normalization='sym', dtype=x.dtype,
            num_nodes=x.size(self.node_dim)
        )

        # L_tilde = L - I
        edge_index_tilde, norm_tilde = add_self_loops(
            edge_index1, norm1, fill_value=-1.0,
            num_nodes=x.size(self.node_dim)
        )

        Tx_0 = x
        Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)

        out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1

        for i in range(2, self.K + 1):
            Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
            Tx_2 = 2 * Tx_2 - Tx_0
            out = out + coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, temp={self.temp})'


# ===============================
# ChebNetII Encoder
# ===============================
# class ChebnetII_Encoder(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, activation, K: int = 2, num_layers: int = 2):
#         """
#         Args:
#             in_channels: number of input features
#             out_channels: number of output features
#             activation: activation function (e.g. F.relu)
#             K: Chebyshev polynomial order
#             num_layers: number of propagation layers
#         """
#         super(ChebnetII_Encoder, self).__init__()
#         assert num_layers >= 1
#         self.num_layers = num_layers
#         self.activation = activation

#         # Layer definitions
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(in_channels, 2 * out_channels))
#         for _ in range(1, num_layers - 1):
#             self.layers.append(nn.Linear(2 * out_channels, 2 * out_channels))
#         self.layers.append(nn.Linear(2 * out_channels, out_channels))

#         # Chebnet propagation
#         self.prop = GuidedChebnet_prop(K)

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         # Chebyshev propagation
#         x = self.prop(x, edge_index)

#         # Feed-forward encoding with activations
#         for layer in self.layers:
#             x = self.activation(layer(x))

#         return x

import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, APPNP
import scipy.sparse as sp


class GCN_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, hidden_dim: int = 64, dropout: float = 0.5):
        """
        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension (e.g., num_classes).
            activation (callable): Activation function (e.g., F.relu).
            base_model (nn.Module): Convolution layer type (default: GCNConv).
            k (int): Number of GCN layers.
            hidden_dim (int): Hidden layer dimension.
            dropout (float): Dropout rate between layers.
        """
        super(GCN_Encoder, self).__init__()
        assert k >= 2, "Number of layers k must be >= 2"
        self.k = k
        self.activation = activation
        self.dropout = dropout
        self.base_model = base_model

        # Define GCN layers
        self.conv = nn.ModuleList()
        self.conv.append(base_model(in_channels, hidden_dim))              # First layer
        for _ in range(1, k - 1):
            self.conv.append(base_model(hidden_dim, hidden_dim))           # Hidden layers
        self.conv.append(base_model(hidden_dim, out_channels))             # Output layer

    def reset_parameters(self):
        for layer in self.conv:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
            if i != self.k - 1:  # no activation or dropout after last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sympy
import scipy
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import sympy
import scipy


# ---------------------------------------------------------
# 1. Beta Polynomial Coefficients
# ---------------------------------------------------------
def calculate_theta2(d):
    """Compute Beta polynomial coefficients for i = 0..d."""
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) /
                       (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas

import numpy as np
import sympy
import scipy.special

def calculate_theta2_normalized(d):
    """
    Compute Beta polynomial coefficients for i = 0..d,
    adapted for normalized adjacency spectrum x ∈ [-1, 1].
    
    The Beta kernel is defined as:
        f_i(x) = ((x + 1)/2)^i * (1 - (x + 1)/2)^(d - i) / Beta(i + 1, d + 1 - i)
    """
    thetas = []
    x = sympy.symbols('x')

    for i in range(d + 1):
        # Map x ∈ [-1, 1] → t = (x + 1)/2 ∈ [0, 1]
        t = (x + 1) / 2
        f = sympy.poly(t**i * (1 - t)**(d - i) / scipy.special.beta(i + 1, d + 1 - i))
        coeff = f.all_coeffs()

        # Pad with zeros if needed
        coeff = [float(c) for c in coeff]
        if len(coeff) < d + 1:
            coeff = [0.0] * (d + 1 - len(coeff)) + coeff

        # Reverse order to align with increasing polynomial powers
        inv_coeff = [coeff[d - j] for j in range(d + 1)]
        thetas.append(inv_coeff)

    return np.array(thetas)

def calculate_theta2_normalized_fixed(d):
    """
    Compute Beta polynomial coefficients for x ∈ [-1, 1] directly,
    preserving numerical stability for normalized adjacency matrices.
    """
    import sympy
    import numpy as np
    import scipy.special

    thetas = []
    x = sympy.symbols('x')

    for i in range(d + 1):
        # No remapping; directly define polynomial in [-1, 1]
        # Replace (x/2) with (x + 1)/2 only once if needed, but keep spectrum scaling consistent.
        f = sympy.poly((x / 2 + 0.5) ** i * (1 - (x / 2 + 0.5)) ** (d - i)
                       / scipy.special.beta(i + 1, d + 1 - i))
        coeff = f.all_coeffs()
        coeff = [float(c) for c in coeff]

        # Pad with zeros if needed
        if len(coeff) < d + 1:
            coeff = [0.0] * (d + 1 - len(coeff)) + coeff

        # Reverse to match increasing powers of x
        inv_coeff = [coeff[d - j] for j in range(d + 1)]
        thetas.append(inv_coeff)

    return np.array(thetas)

# ---------------------------------------------------------
# 2. Combined Beta Kernel Convolution Layer
# ---------------------------------------------------------
class CombinedBetaKernelConv(MessagePassing):
    """
    Combined Beta Kernel Convolution Layer.

    Uses *all* Beta polynomial expansions (θ^(0), θ^(1), ..., θ^(d))
    to create a multi-kernel representation:
        h_out = concat_{i=0}^d [Σ_k θ_i,k (I - D^-1/2 A D^-1/2)^k h]

    Similar to BWGNN but generalized to PyTorch Geometric.
    """
    def __init__(self, in_channels, out_channels, d=2,
                 activation=F.leaky_relu, lin=True):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.d = d
        self.activation = activation
        self.use_lin = lin

        # Store all Beta polynomial coefficient sets
        theta_list = calculate_theta2(d)
        self.register_buffer('theta_tensor',
                             torch.tensor(theta_list, dtype=torch.float32))

    def forward(self, x, edge_index):
        # 1. Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        # 2. Compute normalized adjacency (D^-1/2 A D^-1/2)
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        num_nodes = x.size(0)
        A_norm = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))

        # 3. Laplacian
        I = torch.eye(num_nodes, dtype=x.dtype, device=x.device)
        L = I - A_norm

        # 4. Linear projection
        h0 = self.lin(x)

        # 5. Compute all Beta polynomial expansions
        h_all = []
        for theta in self.theta_tensor:
            h = theta[0] * h0
            x_prop = h0.clone()
            for k in range(1, len(theta)):
                x_prop = torch.sparse.mm(L, x_prop)
                h = h + theta[k] * x_prop
            h_all.append(h)

        # 6. Concatenate outputs from all Beta polynomials
        h_concat = torch.cat(h_all, dim=-1)

        # 7. Optional activation
        if self.use_lin and self.activation is not None:
            h_concat = self.activation(h_concat)
        return h_concat


# ---------------------------------------------------------
# 3. Combined Beta Kernel Encoder
# ---------------------------------------------------------
class CombinedBetaKernelEncoder(nn.Module):
    """
    Multi-layer Combined Beta Kernel Encoder for node classification.

    Each layer concatenates outputs from multiple Beta polynomials.
    The dimensionality expands by (d + 1) per layer.
    """
    def __init__(self, in_channels, out_channels,
                 hidden_dim=64, d=2, num_layers=3,
                 activation=F.leaky_relu, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.d = d

        self.layers = nn.ModuleList()
        self.layers.append(CombinedBetaKernelConv(in_channels, hidden_dim, d=d))
        for _ in range(num_layers - 2):
            self.layers.append(CombinedBetaKernelConv(hidden_dim * (d + 1), hidden_dim, d=d))
        self.layers.append(CombinedBetaKernelConv(hidden_dim * (d + 1), out_channels, d=d))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

def calculate_theta2(d):
    """Compute Beta polynomial coefficients (as in BWGNN)."""
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) /
                       (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas

import sympy
import numpy as np
import scipy.special

import sympy
import numpy as np
import scipy.special

import sympy
import numpy as np
import scipy.special

def calculate_theta2_for_Anorm_scaled(d, spectrum_range=0.9):
    """
    Compute Beta polynomial coefficients for A_norm (spectrum in [-spectrum_range, spectrum_range]).
    Each row i contains coefficients for f_i(x) = sum_k coeff[k] * x^k, coeff[0] is constant term.
    Matches the form f_i(x) = comb(d, i) * ((1+x/spectrum_range)/2)^i * ((1-x/spectrum_range)/2)^(d-i)
    """
    x = sympy.symbols('x')
    thetas = []

    for i in range(d + 1):
        # Rescale x by spectrum_range to map [-spectrum_range, spectrum_range] -> [-1,1]
        f = sympy.expand(
            scipy.special.comb(d, i) *
            ((1 + x / spectrum_range) / 2) ** i *
            ((1 - x / spectrum_range) / 2) ** (d - i)
        )

        # Convert to coefficients
        coeff = sympy.Poly(f, x).all_coeffs()  # high->low
        coeff = [float(c) for c in coeff]

        # Pad to length d+1 if necessary
        if len(coeff) < d + 1:
            coeff = [0.0] * (d + 1 - len(coeff)) + coeff

        # Reverse to increasing power order
        inv_coeff = [coeff[d - j] for j in range(d + 1)]
        thetas.append(inv_coeff)

    return np.array(thetas, dtype=np.float32)



# ------------------------------
# 1. Beta Kernel Propagation Layer
# ------------------------------
class BetaKernelConv(MessagePassing):
    """
    Beta Kernel Graph Convolution layer (as used in BWGNN).
    Performs:
        H = sum_{k=0}^K theta_k * (I - D^-1/2 A D^-1/2)^k X
    where theta_k are derived from the Beta polynomial expansion.
    """

    def __init__(self, in_channels, out_channels, K=3, activation=F.leaky_relu, lin=True):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.K = K
        self.activation = activation
        self.use_lin = lin

        # #Precompute beta polynomial coefficients (thetas)
        #theta_list = calculate_theta2(K)
        # #Flatten and normalize
        #self.thetas = torch.tensor(theta_list[0], dtype=torch.float32)
        theta =  compare_fit_panelA(g_low_pass, 'Chebyshev', True, False,self.K+1, 0.0001, 2.00)
        reversed_arr = theta[::-1].copy()
        theta = reversed_arr
        #theta = [4.0, -6.0, 3.0, -0.5]
        print(theta)
        self.thetas = torch.tensor(theta, dtype=torch.float32)

    def forward(self, x, edge_index):
        # 1. Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        # 2. Compute normalization
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 3. Compute normalized adjacency matrix
        num_nodes = x.size(0)
        A_norm = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))

        # 4. Compute Laplacian
        I = torch.eye(num_nodes, dtype=x.dtype, device=x.device)
        L = I - A_norm

        # 5. Apply linear projection
        h = self.lin(x)

        # 6. Polynomial propagation with Beta coefficients
        out = self.thetas[0] * h
        x_prop = h.clone()
        for k in range(1, len(self.thetas)):
            x_prop = torch.sparse.mm(L, x_prop)
            out = out + self.thetas[k] * x_prop

        # 7. Activation (optional)
        if self.use_lin and self.activation is not None:
            out = self.activation(out)

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class BetaKernel_Encoder(nn.Module):
    """
    Multi-layer Beta Kernel Encoder for node classification.
    """

    def __init__(self, in_channels, out_channels,
                 hidden_dim=64, K=3, num_layers=3,
                 activation=F.leaky_relu, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(BetaKernelConv(in_channels, hidden_dim, K=K))
        for _ in range(num_layers - 2):
            self.layers.append(BetaKernelConv(hidden_dim, hidden_dim, K=K))
        self.layers.append(BetaKernelConv(hidden_dim, out_channels, K=K))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebpts1
from scipy.linalg import inv
# -------------------------------
# Chebyshev nodes
# -------------------------------
def ChebyNodes(N, a, b):
    x = np.cos(np.pi * (2*np.arange(1, N+1)-1)/(2*N))
    return 0.5*(b-a)*x + 0.5*(b+a)

# -------------------------------
def filter_jackson(d):
    N = len(d)
    j = np.arange(N)
    g = (1 - j/N) * d
    return g

# # -------------------------------
# # Parameters
# # -------------------------------
# N = 7
# a = -.95
# b = .95

# # Chebyshev nodes
# interp_t = ChebyNodes(N, a, b)
# interp_y = np.array([f0_beta(x) for x in interp_t])

# # -------------------------------
# # Vandermonde interpolation
# # -------------------------------
# VandermondeMatrix = np.vander(interp_t, N, increasing=True)
# weights = inv(VandermondeMatrix) @ interp_y

# def polynomial_interp(x):
#     powers = np.arange(N)
#     return np.dot(x**powers, weights)

# # -------------------------------
# # Arnoldi interpolation
# # -------------------------------
# t = np.arange(a, b, 0.0001)
# arnoldiCoeff, H = polyfitA(interp_t, interp_y, N-1)
# Ay, _ = polyvalA(arnoldiCoeff, H, t)
# arnoldiCoeff = filter_jackson(arnoldiCoeff)

# y_true = np.array([f0_beta(xi) for xi in t])
# z_vander = np.array([polynomial_interp(xi) for xi in t])

# # -------------------------------
# # Subplots: Arnoldi + Vandermonde
# # -------------------------------
# c1 = [0.83, 0.14, 0.14]
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# # (1) Arnoldi interpolation
# axes[0, 0].plot(t, y_true, color='b', linewidth=2, label='True g(x)')
# axes[0, 0].plot(interp_t, interp_y, 'o', color='gray', label='Chebyshev Nodes')
# axes[0, 0].plot(t, Ay, color=c1, linewidth=2, label='Arnoldi p(x)')
# axes[0, 0].set_title('Arnoldi Interpolation', fontsize=13, fontweight='bold')
# axes[0, 0].set_xlabel('x')
# axes[0, 0].set_ylabel('g(x)')
# axes[0, 0].legend()
# axes[0, 0].grid(True)

# # (2) Arnoldi Error
# axes[0, 1].plot(t, np.abs(y_true - Ay), color='b', linewidth=2)
# axes[0, 1].set_title('Arnoldi Absolute Error', fontsize=13, fontweight='bold')
# axes[0, 1].set_xlabel('x')
# axes[0, 1].set_ylabel('|g - p_A|')
# axes[0, 1].grid(True)

# # (3) Vandermonde interpolation
# axes[1, 0].plot(t, y_true, color='b', linewidth=2, label='True g(x)')
# axes[1, 0].plot(interp_t, interp_y, 'o', color='gray', label='Chebyshev Nodes')
# axes[1, 0].plot(t, z_vander, color='green', linewidth=2, label='Vandermonde p(x)')
# axes[1, 0].set_title('Vandermonde Interpolation', fontsize=13, fontweight='bold')
# axes[1, 0].set_xlabel('x')
# axes[1, 0].set_ylabel('g(x)')
# axes[1, 0].legend()
# axes[1, 0].grid(True)

# # (4) Vandermonde Error
# axes[1, 1].plot(t, np.abs(y_true - z_vander), color='green', linewidth=2)
# axes[1, 1].set_title('Vandermonde Absolute Error', fontsize=13, fontweight='bold')
# axes[1, 1].set_xlabel('x')
# axes[1, 1].set_ylabel('|g - p_V|')
# axes[1, 1].grid(True)

# plt.tight_layout()
# plt.show()
# ------------------------------
# 1. Heat Kernel Propagation Layer
# ------------------------------
class HeatKernelConv(MessagePassing):
    """
    Heat Kernel Convolution layer (truncated series approximation).
    Performs:
        H = sum_{k=0}^K c_k * S^k X
    where S = D^{-1/2} A D^{-1/2}, and c_k = e^{-t} * t^k / k!
    """
    def __init__(self, in_channels, out_channels, K=5, t=3.0, method = 'Heat', kernel = 'Heat_I'):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.K = K
        self.t = float(t)
        self.method = method
        self.simple = True
        self.learn = True
        self.Threeterm = False
        self.kernel = kernel
        assert method in ['Heat', 'PPR', 'RW', 'Complex', 'Beta']
        if method == 'Heat':
        # Precompute coefficients
            coeffs = [math.exp(-self.t) * (self.t ** k) / math.factorial(k) for k in range(K + 1)]
            #coeffs = [math.exp(-self.t) * (self.t ** k) / math.factorial(k) for k in range(K + 1)]
        elif method == 'Diffusion':
            # (I + βL)^-1 ≈ (1/(1+β)) * sum_{k=0}^K (β/(1+β))^k S^k
            beta = 10.0  # very homophilic when large
            coeffs = [(1 / (1 + beta)) * ((beta / (1 + beta)) ** k) for k in range(K + 1)]
        elif method == 'PPR':
            #print("PPR is running")
            alpha = 0.9
            coeffs = [( alpha** k) for k in range(K + 1)]
        elif method == 'Beta':
            # theta_list = calculate_theta2(self.K)
            # coeffs = theta_list[0]
            theta_list= calculate_theta2_for_Anorm_scaled(self.K)
            coeffs = theta_list[0]
            reversed_arr = coeffs[::-1].copy()
            coeffs = reversed_arr
            print(coeffs )
            self.simple = True
        elif method == 'RW':
            lower = -.9
            upper = .9
            assert kernel in ['Heat_I', 'Heat_A', 'RWR_T', 'RWR_I', 'Beta_D','Random']
            print("Kernel is ", kernel)
            if kernel == 'Heat_I':
                coeffs =  compare_fit_panelA(g_low_heat, 'Chebyshev', False, self.Threeterm,self.K+1, lower, upper) # Vadermonde indicator is False here
            elif kernel == 'RWR_I':
                coeffs =  compare_fit_panelA(g_0, 'Chebyshev', False, self.Threeterm,self.K+1, lower, upper) # Vadermonde indicator is False here               
            elif kernel == 'Heat_A':
                coeffs =  compare_fit_panelA(g_low_heat, 'Chebyshev', True, self.Threeterm,self.K+1, lower, upper) # Vadermonde indicator is True here
            elif kernel == 'RWR_T':
                alpha = 0.9
                coeffs = [( alpha** k) for k in range(K + 1)]            
            elif kernel == 'Random':
                coeffs = np.random.rand(K + 1)
                coeffs = coeffs / coeffs.sum()
                #coeffs =  compare_fit_panelA(g_0, 'Chebyshev', True, self.Threeterm,self.K+1, lower, upper) # Vadermonde indicator is True here
            elif kernel == 'Beta_D':
                coeffs =  compare_fit_panelA(f0_beta, 'Chebyshev', True, self.Threeterm,self.K+1, lower, upper) # Vadermonde indicator is True here
            
            reversed_arr = coeffs[::-1].copy()
            coeffs = reversed_arr

            print(coeffs)
            if (self.Threeterm):
                print("Do not change coeffs")
            else:
                if (self.simple):
                    # reversed_arr = coeffs[::-1].copy()
                    # coeffs = reversed_arr
                    print("Do not change coeffs")
                else: 
                    print("Laplacian do not change")

            
            
        print(coeffs)   
        self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float32))
        if(self.learn):
            self.coeffs = nn.Parameter(self.coeffs)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 5. Build normalized adjacency matrix A_norm as a sparse matrix
        num_nodes = x.size(0)
        A_norm = torch.sparse_coo_tensor(
            edge_index,
            norm,
            (num_nodes, num_nodes)
        )
        
        # 6. Create identity matrix I
        #I = torch.eye(num_nodes, dtype=x.dtype)
		I = torch.eye(A_norm.size(0), device=A_norm.device)
        
        # 7. Compute Laplacian: L = I - A_norm
        L = I - A_norm

        x = self.lin(x)
        
        if(self.Threeterm):
            Tx_0 = x
            if(self.simple):
               Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            else:
                Tx_1 = torch.matmul(L, x)
            
            out=self.coeffs[0]*Tx_0+self.coeffs[1]*Tx_1
            
            for i in range(2, self.K+1):
                if(self.simple ):
                    Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
                else:
                    Tx_2 = torch.matmul(L, Tx_1)
                
                Tx_2 = 2 * Tx_2 - Tx_0
                out = out + self.coeffs[i] * Tx_2
                Tx_0, Tx_1 = Tx_1, Tx_2
            
        else:
        
            
            out = self.coeffs[0] * x
            x_prop = x.clone()
            for k in range(1, self.K + 1):
                if(self.simple):
                    x_prop = torch.matmul(A_norm, x_prop)    
                else:
                    x_prop = torch.matmul(L, x_prop)
                #out = out + self.coeffs[k] * x_prop + (1- self.coeffs[k])*x
                out = out + self.coeffs[k] * x_prop
        
        return out
    
    # def forward(self, x, edge_index): 
    #     # ---------------------------------------------------------
    #     # 1. Normalize adjacency with self-loops
    #     # ---------------------------------------------------------
    #     edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    #     row, col = edge_index
    #     deg = degree(col, x.size(0), dtype=x.dtype)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
    #     num_nodes = x.size(0)
    #     A_norm = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))
    
    #     # ---------------------------------------------------------
    #     # 2. Laplacian (used if self.simple == False)
    #     # ---------------------------------------------------------
    #     I = torch.eye(num_nodes, dtype=x.dtype, device=x.device)
    #     L = I - A_norm
    
    #     # ---------------------------------------------------------
    #     # 3. Apply linear transformation
    #     # ---------------------------------------------------------
    #     x = self.lin(x)
        
    #     # ---------------------------------------------------------
    #     # 4. Arnoldi/Chebyshev-like Three-Term Recurrence
    #     # ---------------------------------------------------------
    #     if self.Threeterm:
    #         Tx_0 = x
    #         if self.simple:
    #             Tx_1 = torch.sparse.mm(A_norm, x)
    #         else:
    #             Tx_1 = torch.matmul(L, x)
            
    #         out = self.coeffs[0] * Tx_0 + self.coeffs[1] * Tx_1
            
    #         for i in range(2, self.K + 1):
    #             if self.simple:
    #                 Tx_2 = torch.sparse.mm(A_norm, Tx_1)
    #             else:
    #                 Tx_2 = torch.matmul(L, Tx_1)
                
    #             Tx_2 = 2 * Tx_2 - Tx_0
    #             out = out + self.coeffs[i] * Tx_2
    #             Tx_0, Tx_1 = Tx_1, Tx_2
    
    #     # ---------------------------------------------------------
    #     # 5. Beta Polynomial Expansion (for A_norm)
    #     # ---------------------------------------------------------
    #     else:
    #         out = self.coeffs[0] * x
    #         x_prop = x.clone()
    
    #         # A_norm propagation path
    #         if self.simple:
    #             for k in range(1, self.K + 1):
    #                 x_prop = torch.sparse.mm(A_norm, x_prop)
    
    #                 # Use learned coefficients from calculate_theta2_for_Anorm
    #                 # Ensure spectrum alignment: A_norm ∈ [0,1], not [-1,1]
    #                 out = out + self.coeffs[k] * (2 * x_prop - x)
            
    #         # Laplacian propagation path
    #         else:
    #             for k in range(1, self.K + 1):
    #                 x_prop = torch.matmul(L, x_prop)
    #                 out = out + self.coeffs[k] * x_prop
    
    #     return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# ------------------------------
# 2. Heat Kernel Encoder (GCN-style)
# ------------------------------
class HeatKernel_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=F.relu,
                 K: int = 2, hidden_dim: int = 64, dropout: float = 0.5,
                 heat_K: int = 5, t: float = 3.0, method: str = 'Heat', kernel: str = 'Heat_I'):
        """
        Heat Kernel Encoder with multiple layers, similar to GPRGCN_Encoder.

        Args:
            in_channels (int): Input feature dimension
            out_channels (int): Output feature dimension
            activation (callable): Activation function
            K (int): Number of layers
            hidden_dim (int): Hidden layer dimension
            dropout (float): Dropout rate
            heat_K (int): Truncation order for heat kernel
            t (float): Heat diffusion time
        """
        super().__init__()
        assert K >= 2, "Number of layers K must be >= 2"
        self.k = K
        self.activation = activation
        self.dropout = dropout

        # Define heat kernel layers
        self.conv = nn.ModuleList()
        self.conv.append(HeatKernelConv(in_channels, hidden_dim, K=heat_K, t=t, method = method,kernel=kernel))  # first layer
        for _ in range(1, K - 1):
            self.conv.append(HeatKernelConv(hidden_dim, hidden_dim, K=heat_K, t=t, method = method,kernel=kernel))  # hidden layers
        self.conv.append(HeatKernelConv(hidden_dim, out_channels, K=heat_K, t=t, method = method,kernel=kernel))  # output layer

    def reset_parameters(self):
        for layer in self.conv:
            if hasattr(layer, 'lin'):
                layer.lin.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
            if i != self.k - 1:  # no activation or dropout after last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

#---------------------------------
# This is for monomic polynonomial
#---------------------------------
# ------------------------------
# 1. Heat Kernel Propagation Layer
# ------------------------------
class MonoConv(MessagePassing):
    """
    Heat Kernel Convolution layer (truncated series approximation).
    Performs:
        H = sum_{k=0}^K c_k * S^k X
    where S = D^{-1/2} A D^{-1/2}, and c_k = e^{-t} * t^k / k!
    """
    def __init__(self, in_channels, out_channels, K=5, t=3.0, Init = 'Monomial', lower = 0.0001, upper = 2, nameFunc = 'g_0'):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.K = K
        self.t = float(t)
        self.Init = Init
        self.lower = lower
        self.upper = upper
        Vandermonde = False
        Threeterm = True
        self.homophily = True
        
        assert Init in ['Monomial', 'Chebyshev', 'Legendre', 'Jacobi', 'PPR','SChebyshev']
        if Init == 'Monomial':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            #x = m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
            if(nameFunc == 'g_0'):
                self.coeffs =  compare_fit_panelA(g_0, Init, Vandermonde, Threeterm,self.K, self.lower, self.upper) # m_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
            elif(nameFunc == 'g_1'):
                self.coeffs =  compare_fit_panelA(g_1, Init, Vandermonde,Threeterm,self.K, self.lower, self.upper) 
            elif(nameFunc == 'g_2'):
                self.coeffs =  compare_fit_panelA(g_2,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs =  compare_fit_panelA(g_3,Init,Vandermonde, Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
               self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            l = [i for i in range (1, len(self.coeffs)+1) ]
            self.coeffs = filter_jackson(self.coeffs)
            TEMP = self.coeffs
            
            # TEMP = p_polynomial_zeros(self.K)
            # TEMP = j_polynomial_zeros(self.K,0,1)
        elif Init == 'Chebyshev':
            # PPR-like
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, Vandermonde,Threeterm, self.K,self.lower, self.upper) 
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
                self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init, Vandermonde,Threeterm,self.K)
            l = [i for i in range (1, len(self.coeffs)+1) ]
            #self.coeffs = np.divide(self.coeffs, l)
            self.coeffs = filter_jackson(self.coeffs)
            #self.coeffs = np.divide(self.coeffs, self.division)
            
            TEMP = self.coeffs
            #TEMP = t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)
        elif Init == 'Legendre':
            #TEMP = p_polynomial_zeros(self.K)
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, Vandermonde,Threeterm, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K) 
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, Vandermonde,Threeterm,self.K,self.lower, self.upper) #p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,Vandermonde,Threeterm,  self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm,self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm,self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
                self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,self.K,self.lower, self.upper)
            l = [i for i in range (1, len(self.coeffs)+1)]
            self.coeffs = filter_jackson(self.coeffs)
            #self.coeffs = np.divide(self.coeffs, l)
            #self.coeffs = np.divide(self.coeffs, self.division)
            
            TEMP = self.coeffs
        elif Init == 'Jacobi':
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, Vandermonde, Threeterm, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K) 
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, Vandermonde,Threeterm, self.K,self.lower, self.upper) #p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,Vandermonde, Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,Vandermonde, Threeterm, self.K,self.lower, self.upper)#p_polynomial_zeros(self.K)
            elif(nameFunc == 'g_4'):
                self.coeffs = compare_fit_panelA(g_4,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_rejection'):
                self.coeffs = compare_fit_panelA(g_band_rejection,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)#t_polynomial_zeros(-(self.alpha), (self.alpha), self.K)#
            elif(nameFunc == 'g_band_pass'):
                self.coeffs = compare_fit_panelA(g_band_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_low_pass'):
                self.coeffs = compare_fit_panelA(g_low_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_high_pass'):
                self.coeffs = compare_fit_panelA(g_high_pass,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            elif(nameFunc == 'g_comb'):
                self.coeffs = compare_fit_panelA(g_comb,Init,Vandermonde,Threeterm, self.K,self.lower, self.upper)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,self.K)
            l = [i for i in range (1, len(self.coeffs)+1) ]
            #self.coeffs = np.divide(self.coeffs, l)
            
            #self.coeffs = np.divide(self.coeffs, self.division)
            TEMP = self.coeffs
            #TEMP = j_polynomial_zeros(self.K,0,1)
        elif Init == 'SChebyshev':
            #TEMP = s_polynomial_zeros(self.K)
            if(nameFunc == 'g_0'):
                self.coeffs = compare_fit_panelA(g_0, Init, self.K)
            elif(nameFunc == 'g_1'):
                self.coeffs = compare_fit_panelA(g_1, Init, self.K) 
            elif(nameFunc == 'g_2'):
                self.coeffs = compare_fit_panelA(g_2,Init,self.K)
            elif(nameFunc == 'g_3'):
                self.coeffs = compare_fit_panelA(g_3,Init,self.K)
            else:
                self.coeffs = compare_fit_panelA(g_fullRWR,Init,self.K)
            TEMP = self.coeffs
        elif Init == 'PPR':
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))
        coeffs = self.coeffs
        # Precompute coefficients
        #coeffs = [math.exp(-self.t) * (self.t ** k) / math.factorial(k) for k in range(K + 1)]
        #self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x, edge_index):
            
            coe_tmp = torch.from_numpy(self.coeffs)
    
        # mdic = {"A": A}
        # savemat("Cora.mat", mdic)
        #coe_tmp = torch.flip(coe_tmp, dims=(0,))
            coe = coe_tmp.clone()
            coe = self.coeffs
        
        # for i in range(self.K + 1):
        #     coe[i] = coe_tmp[0] * cheby(i, math.cos((self.K + 0.5) * math.pi / (self.K + 1)))
        #     for j in range(1, self.K + 1):
        #         x_j = math.cos((self.K - j + 0.5) * math.pi / (self.K + 1))
        #         coe[i] = coe[i] + coe_tmp[j] * cheby(i, x_j)
        #     coe[i] = 2 * coe[i] / (self.K + 1)
        
        # L=I-D^(-0.5)AD^(-0.5)
            edge_weight=None
            edge_index, norm = gcn_norm(edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                               num_nodes=x.size(self.node_dim))
            
            # L_tilde=L-I
            edge_index_tilde, norm_tilde = add_self_loops(edge_index1, norm1, fill_value=-1.0,
                                                          num_nodes=x.size(self.node_dim))
            
            
            # homophily = False
            
            # if (homophily):
            #     Tx_1=self.propagate(edge_index,x=x,norm=norm,size=None)
            # else:
            #     Tx_1=self.propagate(edge_index1,x=x,norm=norm1,size=None)
            # out=coe[0]*Tx_0+coe[1]*Tx_1
            # #out = coe[0] / 2 * Tx_0 + coe[1] * Tx_1
            
            # for i in range(2, self.K-1):
            #     if (homophily):
            #         Tx_2=self.propagate(edge_index,x=Tx_1,norm=norm,size=None)
            #     else:
            #         Tx_2=self.propagate(edge_index1,x=Tx_1,norm=norm1,size=None)
            #     Tx_2 = 2 * Tx_2 - Tx_0
            #     out = out + coe[i] * Tx_2
            #     Tx_0, Tx_1 = Tx_1, Tx_2
            
            # return out
            self.homophily = True
            x = self.lin(x)
            Tx_0 = x
            if(self.homophily):
               Tx_1 = self.propagate(edge_index_tilde, x=x, norm=norm_tilde, size=None)
            else:
                Tx_1 = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            
            out=coe[0]*Tx_0+coe[1]*Tx_1
            
            for i in range(2, self.K-1):
                if(self.homophily ):
                    Tx_2 = self.propagate(edge_index_tilde, x=Tx_1, norm=norm_tilde, size=None)
                else:
                    Tx_2 = self.propagate(edge_index1, x=Tx_1, norm=norm1, size=None)
                
                Tx_2 = 2 * Tx_2 - Tx_0
                out = out + coe[i] * Tx_2
                Tx_0, Tx_1 = Tx_1, Tx_2
            
            return out
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # # 5. Build normalized adjacency matrix A_norm as a sparse matrix
        # num_nodes = x.size(0)
        # A_norm = torch.sparse_coo_tensor(
        #     edge_index,
        #     norm,
        #     (num_nodes, num_nodes)
        # )
        
        # # 6. Create identity matrix I
        # I = torch.eye(num_nodes, dtype=x.dtype)
        
        # # 7. Compute Laplacian: L = I - A_norm
        # L = I - A_norm

        # x = self.lin(x)
        # out = self.coeffs[0] * x
        # x_prop = x.clone()
        # self.homophily = True
        # #x_prop = self.coeffs[self.K-1]*x_propband
        # for k in range(1, self.K):
        #     if (self.homophily):
        #         x_prop = torch.matmul(A_norm, x_prop)
        #     else:
        #         x_prop = torch.matmul(L, x_prop)

        #     out = out + self.coeffs[k] * x_prop
        # return out
    # def forward(self, x, edge_index, edge_weight=None):
    #     edge_index, norm = gcn_norm(
    #         edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
    #     edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
    #     #edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))
    #     #2I-L
    #     edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))
    #     hidden = self.coeffs[self.K-1]*x
    #     #hidden = x*(self.temp[0])
    #     for k in range(self.K-2,-1,-1):
    #         if (self.homophily):
    #             x = self.propagate(edge_index, x=x, norm=norm)             
    #         else:       
    #             x = self.propagate(edge_index1, x=x, norm=norm1)
    #         gamma = self.coeffs[k]
            
    #         x = x + gamma*hidden
    #     return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# ------------------------------
# 2. Heat Kernel Encoder (GCN-style)
# ------------------------------
class Mono_GEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=F.relu,
                 K: int = 2, hidden_dim: int = 64, dropout: float = 0.5,
                 heat_K: int = 5, t: float = 3.0, Init: str = 'Monomial', lower: float = 0.0001, upper: float = 2.0, nameFunc:str = 'g_0'):
        """
        Heat Kernel Encoder with multiple layers, similar to GCN_Encoder.

        Args:
            in_channels (int): Input feature dimension
            out_channels (int): Output feature dimension
            activation (callable): Activation function
            K (int): Number of layers
            hidden_dim (int): Hidden layer dimension
            dropout (float): Dropout rate
            heat_K (int): Truncation order for heat kernel
            t (float): Heat diffusion time
        """
        super().__init__()
        assert K >= 2, "Number of layers K must be >= 2"
        self.k = K
        self.activation = activation
        self.dropout = dropout

        # Define heat kernel layers
        self.conv = nn.ModuleList()
        self.conv.append(MonoConv(in_channels, hidden_dim, K=heat_K, t=t, Init = 'Monomial', lower = 0.0001, upper = 2, nameFunc = 'g_0'))  # first layer
        for _ in range(1, K - 1):
            self.conv.append(MonoConv(hidden_dim, hidden_dim, K=heat_K, t=t,Init = 'Monomial', lower = 0.0001, upper = 2, nameFunc = 'g_0'))  # hidden layers
        self.conv.append(MonoConv(hidden_dim, out_channels, K=heat_K, t=t,Init = 'Monomial', lower = 0.0001, upper = 2, nameFunc = 'g_0'))  # output layer

    def reset_parameters(self):
        for layer in self.conv:
            if hasattr(layer, 'lin'):
                layer.lin.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
            if i != self.k - 1:  # no activation or dropout after last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

class ChebnetII_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model, k: int = 2, hidden_dim: int = 64,
                 dropout: float = 0.5, dprate: float = 0.0, residual: bool = True):
        """
        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension (e.g., num_classes).
            activation (callable): Non-linear activation (e.g., F.relu).
            base_model (nn.Module): ChebnetII_prop or similar propagation class.
            k (int): Number of layers (Linear + ChebnetII per layer).
            hidden_dim (int): Hidden dimension.
            dropout (float): Dropout between layers.
            dprate (float): Additional dropout before final layer.
            residual (bool): Use residual/skip connections.
        """
        super(ChebnetII_Encoder, self).__init__()
        assert k >= 2, "Number of layers k must be >= 2"
        self.k = k
        self.activation = activation
        self.dropout = dropout
        self.dprate = dprate
        self.residual = residual

        # Linear transformation layers
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(in_channels, hidden_dim))
        for _ in range(1, k - 1):
            self.lin_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.lin_layers.append(nn.Linear(hidden_dim, out_channels))

        # Chebyshev propagation layers
        self.prop_layers = nn.ModuleList()
        for _ in range(k):
            self.prop_layers.append(base_model())

        # Optional linear projection for residual shape matching
        if residual:
            self.res_proj = nn.ModuleList()
            self.res_proj.append(nn.Identity() if in_channels == hidden_dim else nn.Linear(in_channels, hidden_dim))
            for _ in range(1, k - 1):
                self.res_proj.append(nn.Identity())
            self.res_proj.append(nn.Identity() if hidden_dim == out_channels else nn.Linear(hidden_dim, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lin_layers:
            lin.reset_parameters()
        for prop in self.prop_layers:
            prop.reset_parameters()
        if self.residual:
            for proj in self.res_proj:
                if hasattr(proj, "reset_parameters"):
                    proj.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = x  # initial input for residual
        for i in range(self.k):
            out = self.lin_layers[i](x)

            if i != self.k - 1:
                out = self.activation(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
            elif self.dprate > 0:
                out = F.dropout(out, p=self.dprate, training=self.training)

            out = self.prop_layers[i](out, edge_index)

            if self.residual:
                res = self.res_proj[i](h)
                out = out + res  # skip connection
            h = out  # update residual input for next layer
            x = out  # update input

        return F.log_softmax(x, dim=1)
    
class APPNP_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 hidden_dim: int = 64, dropout: float = 0.5,
                 K: int = 10, alpha: float = 0.1):
        """
        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension (e.g., num_classes).
            activation (callable): Activation function (e.g., F.relu).
            hidden_dim (int): Hidden layer size for the MLP.
            dropout (float): Dropout rate.
            K (int): Number of propagation steps in APPNP.
            alpha (float): Teleport (personalization) probability.
        """
        super(APPNP_Encoder, self).__init__()
        self.activation = activation
        self.dropout = dropout

        # MLP part (feature transformation)
        self.lin1 = nn.Linear(in_channels, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_channels)

        # APPNP propagation layer
        self.prop = APPNP(K=K, alpha=alpha)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Feature transformation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.lin1(x))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # APPNP propagation
        x = self.prop(x, edge_index)

        return F.log_softmax(x, dim=1)

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# -----------------------------------
# 1. GPRGNN Propagation Layer
# -----------------------------------
class GPRProp(MessagePassing):
    """
    Generalized PageRank propagation with learnable coefficients gamma.
    """
    def __init__(self, K, alpha=0.1, Init='PPR'):
        super().__init__(aggr='add')
        self.K = K
        self.alpha = alpha
        self.Init = Init
        self.gamma = nn.Parameter(torch.Tensor(K + 1))  # learnable coefficients
        self.reset_parameters()

    def reset_parameters(self):
        if self.Init == 'PPR':
            temp = self.alpha * (1 - self.alpha) ** torch.arange(self.K + 1)
            temp[-1] = (1 - self.alpha) ** self.K
            self.gamma.data = temp
        elif self.Init == 'NPPR':
            temp = (1 - self.alpha) ** torch.arange(self.K + 1)
            temp = temp / temp.sum()
            self.gamma.data = temp
        elif self.Init == 'Random':
            bound = torch.sqrt(torch.tensor(3.0 / (self.K + 1)))
            self.gamma.data = torch.empty(self.K + 1).uniform_(-bound, bound)
        else:  # uniform
            self.gamma.data.fill_(1.0 / (self.K + 1))

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        hidden = x * self.gamma[0]
        x_prop = x.clone()
        for k in range(1, self.K + 1):
            x_prop = self.propagate(edge_index, x=x_prop, norm=norm)
            hidden = hidden + self.gamma[k] * x_prop
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# -----------------------------------
# 2. GPRGNN Encoder (GCN-style)
# -----------------------------------
class GPRGNN_Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=F.relu,
                 K: int = 3, hidden_dim: int = 64, dropout: float = 0.5,
                 alpha: float = 0.1, Init: str = 'PPR'):
        """
        GPRGNN Encoder with learnable propagation coefficients (gamma_k).

        Args:
            in_channels (int): Input feature dimension
            out_channels (int): Output feature dimension
            activation (callable): Activation function
            K (int): Number of propagation steps
            hidden_dim (int): Hidden dimension
            dropout (float): Dropout rate
            alpha (float): Teleport probability (for PPR initialization)
            Init (str): Initialization type for gamma (PPR, NPPR, Random, Uniform)
        """
        super().__init__()
        self.K = K
        self.activation = activation
        self.dropout = dropout

        # MLP before propagation (1-layer)
        self.lin1 = nn.Linear(in_channels, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_channels)

        # GPR propagation
        self.prop = GPRProp(K=K, alpha=alpha, Init=Init)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.lin1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
# GCN encoder
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        #assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            # print(self.conv[i](x, edge_index))
            x = self.activation(self.conv[i](x, edge_index))
            # print(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    # loss definition
    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret



class scPROTEIN_learning(torch.nn.Module):

    def __init__(self, model,device, data, drop_feature_rate_1,drop_feature_rate_2,drop_edge_rate_1,drop_edge_rate_2,
                 learning_rate, weight_decay, num_protos, topology_denoising, num_epochs, alpha, num_changed_edges, seed):
        super(scPROTEIN_learning, self).__init__()
        self.model = model
        self.data = data
        self.device = device
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_protos = num_protos
        self.topology_denoising = topology_denoising
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.seed = seed
        self.num_changed_edges = num_changed_edges
        
    def train(self):
        setup_seed(self.seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            
            optimizer.zero_grad()
            edge_index_1 = dropout_adj(self.data.edge_index, p=self.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(self.data.edge_index, p=self.drop_edge_rate_2)[0]
            x_1 = drop_feature(self.data.x, self.drop_feature_rate_1)
            x_2 = drop_feature(self.data.x, self.drop_feature_rate_2)
            z1 = self.model(x_1, edge_index_1)
            z2 = self.model(x_2, edge_index_2)
            loss_node = self.model.loss(z1, z2, batch_size=0)
            # return loss



            # embedding = test(model, data.x.to(device), data.edge_index.to(device))
            embedding = self.test()
            embedding_cpu = embedding.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=self.num_protos).fit(embedding_cpu)
            label_kmeans = kmeans.labels_
            centers = np.array([np.mean(embedding_cpu[label_kmeans == i,:], axis=0)
                                for i in range(self.num_protos)])
            label_kmeans = label_kmeans[:, np.newaxis]
            proto_norm = get_proto_norm(embedding_cpu, centers,label_kmeans,self.num_protos)
            centers = torch.Tensor(centers).to(self.device)
            label_kmeans = torch.Tensor(label_kmeans).long().to(self.device)
            proto_norm = torch.Tensor(proto_norm).to(self.device)
            loss_proto = get_proto_loss(embedding, centers, label_kmeans, proto_norm)       



            # topology denoising
            if self.topology_denoising:
                with torch.no_grad():
                    embedding_cpu = sp.coo_matrix(embedding_cpu)
                    similarity_matrix = embedding_cpu.dot(embedding_cpu.transpose())
                    similarity_matrix = similarity_matrix.tocoo()
                    coords = list(np.vstack((similarity_matrix.row, similarity_matrix.col)).transpose())
                    coords_tuple = []
                    for i in coords:
                        coords_tuple.append(tuple(i))

                    simi_data = list(similarity_matrix.data)
                    coord_value_dict = {}
                    for i in range(len(coords_tuple)):
                        coord_value_dict[coords_tuple[i]] = simi_data[i]

                    coord_value_dict_wo_diag = {}
                    cnt = 0
                    for key,value in coord_value_dict.items():
                        if key[0] == key[1]:
                            cnt += 1
                        else:
                            coord_value_dict_wo_diag[key] = value

                    coords_wo_diag = list(coord_value_dict_wo_diag.keys())
                    simi_data_wo_diag = np.array(list(coord_value_dict_wo_diag.values()))
                    simi_sort = simi_data_wo_diag.argsort()
                    high_prob_indices = list(simi_sort[-self.num_changed_edges:])
                    low_prob_indices = list(simi_sort[:self.num_changed_edges])
                    high_prob_coords = []
                    low_prob_coords = []
                    for i in high_prob_indices:
                        high_prob_coords.append(coords_wo_diag[i])
                    for i in low_prob_indices:
                        low_prob_coords.append(coords_wo_diag[i])

                    edge_index_now = list(self.data.edge_index.cpu().detach().numpy().T)
                    edge_index_now_list = []
                    for i in edge_index_now:
                        edge_index_now_list.append(tuple(i))
                    cnt_add = 0
                    for i in high_prob_coords:
                        if i not in edge_index_now_list:
                            edge_index_now_list.append(i)
                            cnt_add += 1
                    cnt_remove = 0
                    for i in low_prob_coords:
                        if i in edge_index_now_list:
                            edge_index_now_list.remove(i)
                            cnt_remove += 1

                    edge_index = torch.tensor(np.array(edge_index_now_list).T, dtype=torch.long).to(self.device)
                    self.data.edge_index = edge_index    
            
            
            loss = loss_node + self.alpha*loss_proto
            loss.backward()
            optimizer.step()

            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f} ')
        print("\n=== Learned Beta Kernel Coefficients After Training ===")
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "conv"):
            for i, layer in enumerate(self.model.encoder.conv):
                if isinstance(layer, HeatKernelConv):
                    print(f"Layer {i}: {layer.coeffs.data.cpu().numpy()}")
        else:
            print("No HeatKernelConv layers found in model.")


    def test(self):
        self.model.eval()
        z = self.model(self.data.x, self.data.edge_index)
        return z

    def embedding_generation(self):
        self.model.eval()
        z = self.model(self.data.x, self.data.edge_index)
        return z.cpu().detach().numpy()







