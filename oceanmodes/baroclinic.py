# python 3 forward compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
#
import numpy as np
from scipy import fftpack as fft
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import eigs, inv

def _maybe_truncate_above_topography(z, f):
    """Return truncated versions of z and f if f is masked or nan.
    """
    # checks on shapes of stuff
    if not z.shape == f.shape:
        raise ValueError('z and f must have the same length')

    fm = np.ma.masked_invalid(f)
    if fm.mask.sum()==0:
        return z, f

    # check to make sure the mask is only at the bottom
    #if not fm.mask[-1]:
    #    raise ValueError('topography should be at the bottom of the column')
    # the above check was redundant with this one
    if fm.mask[-1] and np.diff(fm.mask).sum() != 1:
        raise ValueError('topographic mask should be monotonic')

    zout = z[~fm.mask]
    fout = fm.compressed()
    return zout, fout

def neutral_modes_from_N2_profile(z, N2, f0, depth=None, **kwargs):
    """Calculate baroclinic neutral modes from a profile of 
        buoyancy frequency truncated at the bottom.

        Solves the eigenvalue problem

            \frac{d}{dz}\left ( \frac{f_0^2}{N^2} \frac{d \phi}{d z} \right )
            = -\frac{1}{R_d^2} \phi

        With the boundary conditions

            \frac{d \phi}{d z} = 0

        at the top and bottom.

        Parameters
        ----------
        z : array_like
            The depths at which N2 is given. Starts shallow, increases
            positive downward.
        N2 : array_like
            The squared buoyancy frequency (units s^-2). Points below topography
            should be masked or nan.
        f0 : float
            Coriolis parameter.
        depth : float, optional
            Total water column depth. If missing will be inferred from z.
        kwargs : optional
            Additional parameters to pass to scipy.sparse.linalg.eigs for
            eigenvalue computation

        Returns
        -------
        zf : array_like
            The depths at which phi is defined. Different from z.
        R_d : array_like
            deformation radii, sorted descending
        phi : array_like
            vertical modes
    """
    nz_orig = len(z)
    z, N2 = _maybe_truncate_above_topography(z, N2)
    return _neutral_modes_from_N2_profile_raw(
                                    z, N2, f0, depth=depth, **kwargs)
    # does it make sense to re-pad output?

def _neutral_modes_from_N2_profile_raw(z, N2, f0, depth=None, **kwargs):
    nz = len(z)

    ### vertical discretization ###

    # ~~~~~ zf[0]==0, phi[0] ~~~~
    #
    # ----- zc[0], N2[0] --------
    #
    # ----- zf[1], phi[1] -------
    # ...
    # ----- zc[nz-1], N2[nz-1] ---
    #
    # ~~~~~ zf[nz], phi[nz] ~~~~~

    # just for notation's sake
    # (user shouldn't worry about discretization)
    zc = z
    dzc = np.hstack(np.diff(zc))
    # make sure z is increasing
    if not np.all(dzc > 0):
        raise ValueError('z should be monotonically increasing')
    if depth is None:
        depth = z[-1] + dzc[-1]/2
    else:
        if depth <= z[-1]:
            raise ValueError('depth should not be less than maximum z')

    dztop = zc[0]
    dzbot = depth - zc[-1]

    # put the phi points right between the N2 points
    zf = np.hstack([0, 0.5*(zc[1:]+zc[:-1]), depth ])
    dzf = np.diff(zf)

    # We want a matrix representation of the operator such that
    #    g = f0**2 * np.dot(L, f)
    # This can be put in "tridiagonal" form
    # 1) first derivative of f (defined at zf points, maps to zc points)
    #    dfdz[i] = (f[i] - f[i+1]) /  dzf[i]
    # 2) now we are at zc points, so multipy directly by f0^2/N^2
    #    q[i] = dfdz[i] / N2[i]
    # 3) take another derivative to get back to f points
    #    g[i] = (q[i-1] - q[i]) / dzc[i-1]
    # boundary condition is enforced by assuming q = 0 at zf = 0
    #    g[0] = (0 - q[0]) / dztop
    #    g[nz] = (q[nz-1] - 0) / dzbot
    # putting it all together gives
    #    g[i] = ( ( (f[i-1] - f[i]) / (dzf[i-1] * N2[i-1]) )
    #            -( (f[i] - f[i+1]) / (dzf[i] * N2[i])) ) / dzc[i-1]
    # which we can rewrite as
    #    g[i] = ( a*f[i-1] + b*f[i] +c*f[i+1] )
    # where
    #    a = (dzf[i-1] * N2[i-1] * dzc[i-1])**-1
    #    b = -( ((dzf[i-1] * N2[i-1]) + (dzf[i] * N2[i])) * dzc[i-1] )**-1
    #    c = (dzf[i] * N2[i] * dzc[i-1])**-1
    # for the boundary conditions we have
    #    g[0] =  (-f[0] + f[1]) /  (dzf[0] * N2[0] * dztop)
    #    g[nz] = (f[nz-1] - f[nz]) /  (dzf[nz-1] * N2[nz-1] *dzbot)
    # which we can rewrite as
    #    g[0] = (-a*f[0] + a*f[1])
    #           a = (dzf[0] * N2[0])**-1
    #    g[nz] = (b*f[nz-1] - b*f[nz])
    #           b = (dzf[nz-1] * N2[nz-1])**-1

    # now turn all of that into a sparse matrix

    L = lil_matrix((nz+1, nz+1), dtype=np.float64)
    for i in range(1,nz):
        a = (dzf[i-1] * N2[i-1] * dzc[i-1])**-1
        b = -(dzf[i-1] * N2[i-1]* dzc[i-1])**-1 - (dzf[i] * N2[i] * dzc[i-1])**-1
        c = (dzf[i] * N2[i] * dzc[i-1])**-1
        L[i,i-1:i+2] = [a,b,c]
    a = (dzf[0] * N2[0] * dztop)**-1
    L[0,:2] = [-a, a]
    b = (dzf[nz-1] * N2[nz-1] * dzbot)**-1
    L[nz,-2:] = [b, -b]

    # this gets the eignevalues and eigenvectors
    w, v = eigs(L, which='SM')

    # eigs returns complex values. Make sure they are actually real
    tol = 1e-20
    np.testing.assert_allclose(np.imag(v), 0, atol=tol)
    np.testing.assert_allclose(np.imag(w), 0, atol=tol)
    w = np.real(w)
    v = np.real(v)

    # they are often sorted and normalized, but not always
    # so we have to do that here
    j = np.argsort(w)[::-1]
    w = w[j]
    v = v[:,j]
    
    #########
    # w = - 1/(Rd^2 * f0^2)
    #########
    Rd = (-w)**-0.5 / f0

    return zf, Rd, v

def instability_analysis_from_N2_profile(z, N2, f0, beta, Nx, Ny, dx, dy, ubar, vbar, etax, etay, 
                                         sort='LI', num=4, depth=None, **kwargs):
    """Calculate baroclinic neutral modes from a profile of 
        buoyancy frequency truncated at the bottom.
    """
    nz_orig = len(z)
    z, N2 = _maybe_truncate_above_topography(z, N2)
    return _instability_analysis_from_N2_profile_raw(
                                    z, N2, f0, beta, Nx, Ny, dx, dy, ubar, vbar, etax, etay, sort, num, depth=depth, **kwargs)

def _instability_analysis_from_N2_profile_raw(z, N2, f0, beta, Nx, Ny, dx, dy, ubar, vbar, etax, etay, 
                                              sort='LI', num=4, depth=None, **kwargs):
    """Calculate baroclinic unstable modes from a profile of buoyancy frequency.
    
        Solves the eigenvalue problem

            \left ( \frac{\partial}{\partial t} + U \cdot \nabla \right ) \left ( \nabla^2 + \frac{d}{dz} \frac{f_0^2}{N^2} \frac{d \psi}{d z} \right ) \psi
            + u \cdot \nabla Q
            = 0

        With the boundary conditions

            \left ( \frac{\partial}{\partial t} + U \cdot \nabla \right ) f_0 \frac{\partial \psi}{\partial z}
            + u \cdot \nabla \left ( B + N^2 \eta \right )
            = 0

        at the top and bottom.
    
        Parameters
        ----------
        z : array_like
            The depths at which N2 is given. Starts shallow, increases
            positive downward.
        N2 : array_like
            The squared buoyancy frequency (units s^-2). Points below topography
            should be masked or nan.
        f0 : float
            Coriolis parameter.
        Q : array_like
            The background QGPV 
            (\beta y + \left ( \nabla^2 + \frac{d}{dz} \frac{f_0^2}{N^2} \frac{d \Psi}{d z} \right ))
            and Psi is the streamfunction of the background velocity.
        etax, etay : array_like
            The topographic horizontal gradient at the top and bottom.
        B : array_like
            The background buoyancy
        Nx, Ny, dx, dy: float
            Parameters to define the horizontal wavenumber
        sort: string, optional
            Sorts the eigenvalues depending on the given argument
        num: integer, optional
            Number of eigenvalues returned
        depth : float, optional
            Total water column depth. If missing will be inferred from z.
        kwargs : optional
            Additional parameters to pass to scipy.sparse.linalg.eigs for
            eigenvalue computation
    
        Returns
        -------
        zf : array_like
            The depths at which phi is defined. Different from z.
        psi : array_like
            eigenvectors
        omega: array_like
            eigenvalues (imaginary part is the growth rate)
        k, l : array_like
            inverse horizontal wavelengths
    """
    nz = len(z)

    ### vertical discretization ###

    # ~~~~~ zf[0]==0, phi[0] ~~~~
    #
    # ----- zc[0], N2[0] --------
    #
    # ----- zf[1], phi[1] -------
    # ...
    # ---- zc[nz-1], N2[nz-1] ---
    #
    # ~~~~ zf[nz], phi[nz] ~~~~~~

    # just for notation's sake
    # (user shouldn't worry about discretization)
    zc = z
    dzc = np.hstack(np.diff(zc))
    # make sure z is increasing
    if not np.all(dzc > 0):
        raise ValueError('z should be monotonically increasing')
    if depth is None:
        depth = z[-1] + dzc[-1]/2
    else:
        if depth <= z[-1]:
            raise ValueError('depth should not be less than maximum z')

    dztop = zc[0]
    dzbot = depth - zc[-1]

    # put the phi points right between the N2 points
    zf = np.hstack([0, 0.5*(zc[1:]+zc[:-1]), depth ])
    dzf = np.diff(zf)
    
    # We want a matrix representation of the operator such that
    #    np.dot(L, f) = g = np.dot(G, f)
    # This can be put in "tridiagonal" form
    # 1) first derivative of f (defined at zf points, maps to zc points)
    #    dfdz[i] = (f[i] - f[i+1]) /  dzf[i]
    # 2) now we are at zc points, so multipy directly by f0^2/N^2
    #    q[i] = dfdz[i] / N2[i]
    # 3) take another derivative to get back to f points
    #    P[i] = k \frac{\partial Q[i]}{\partial y} - l \frac{\partial Q[i]}{\partial x}
    #    g[i] = (kU + lV) * ((q[i-1] - q[i]) / dzc[i-1] - K**2 * q[i]) + P[i] * f[i]
    # boundary condition is enforced by assuming q = 0 at zf = 0
    #    S[i] = k * (\frac{\partial U[i]}{\partial z} - N2[i]/f_0 \frac{\partial \eta[i]}{\partial y}) + l * (\frac{\partial V[0]}{\partial z} - N2[i]/f_0 \frac{\partial \eta[i]}{\partial x})
    #    g[0] = ((0 - q[0]) / dztop - S[0] * f[0]
    #    g[nz] = ((q[nz-1] - 0) / dzbot - S[nz] * f[nz]
    # putting it all together gives
    #    g[i] = (kU + lV) * ( ( ( (f[i-1] - f[i]) / (dzf[i-1] * N2[i-1]) )
    #            -( (f[i] - f[i+1]) / (dzf[i] * N2[i])) ) / dzc[i-1] - K**2 ) + P[i] * f[i]
    # which we can rewrite as
    #    g[i] = ( a*f[i-1] + b*f[i] +c*f[i+1] )
    # where
    #    a = (kU[i] + lV[i]) * ( (dzf[i-1] * N2[i-1] * dzc[i-1])**-1 - K**2 ) + P[i]
    #    b = (kU[i] + lV[i]) * ( -( ((dzf[i-1] * N2[i-1]) + (dzf[i] * N2[i])) * dzc[i-1] )**-1 - K**2 ) + P[i]
    #    c = (kU[i] + lV[i]) * ( (dzf[i] * N2[i] * dzc[i-1])**-1 - K**2 ) + P[i]
    # for the boundary conditions we have
    #    g[0] =  (kU[i] + lV[i]) * (-f[0] + f[1]) /  (dzf[0] * N2[0] * dztop) - S[0] * f[0]
    #    g[nz] = (kU[nz] + lV[nz]) * (f[nz-1] - f[nz]) /  (dzf[nz-1] * N2[nz-1] * dzbot) - S[nz] * f[nz]
    # which we can rewrite as
    #    g[0] = (-a*f[0] + a*f[1])
    #    g[nz] = (b*f[nz-1] - b*f[nz])
    # where
    #    a = (kU[i] + lV[i]) * (dzf[0] * N2[0] * dztop)**-1  - S[0] 
    #    b = (kU[nz] + lV[nz]) * (dzf[nz-1] * N2[nz-1])**-1 * dzbot) - S[nz] 
    
    k = fft.fftshift( fft.fftfreq(Nx, dx) )
    l = fft.fftshift( fft.fftfreq(Ny, dy) )
    
    for j in range(len(l)):
        for i in range(len(k)):
        
            #print i, j, k[i], l[j]
        
            L = lil_matrix((nz+1, nz+1), dtype=np.float64)
            G = L.copy()
                
            for n in range(nz+1):
                    
                R = ( k[i]*ubar[n] + l[j]*vbar[n] )
#             print R

                if n == 0:
    
                    D = .5 * dzf[n]**-1
                    S = ( k[i] * ( (ubar[n+1]-ubar[n])*D - N2[n]/f0 * etay[0] )
                                        + l[j] * ( (vbar[n+1]-vbar[n])*D + N2[n]/f0 * etax[0] ) )
#                 print D, S

                    L[n, n] = R * ( -D ) - S
                    L[n, n+1] = R * D - S
                    G[n, n] = - D
                    G[n, n+1] = D
    
                elif n > 0 and n < nz:
    
                    K2 = k[i]**2 + l[j]**2
                    bf = f0**2 * dzc[n-1]**-1
                    b_1 = N2[n-1] * dzf[n-1]
                    b = N2[n] * dzf[n]
                    B_1 = bf * b_1**-1 - K2
                    B = - bf * ( b**-1 + b_1**-1 ) - K2
                    Bt1 = bf * b**-1 - K2
    
                    N2Z = (N2[n]*dzf[n])**-1
                    N2Z_1 = (N2[n-1]*dzf[n-1])**-1
                    P = ( k[i] * ( beta - bf * ( ubar[n+1] * N2Z
                                                                - (N2Z + N2Z_1) * ubar[n]
                                                                + N2Z_1 * ubar[n-1] ) )
                         - l[j] * bf * ( vbar[n+1] * N2Z 
                                                           - (N2Z + N2Z_1) * vbar[n]
                                                           + N2Z_1 * vbar[n-1] )
                          )
#                 print B_1, B, Bt1, P

                    L[n, n-1] = R * B_1 + P
                    L[n, n] = R * B + P
                    L[n, n+1] = R * Bt1 + P
                    G[n, n-1] = B_1
                    G[n, n] = B
                    G[n, n+1] = Bt1

                else:

                    D = .5 * dzf[n-1]**-1
                    S = ( k[i] * ( (ubar[n]-ubar[n-1])*D - N2[n-1]/f0 * etay[1] )
                                        + l[j] * ( (vbar[n]-vbar[n-1])*D + N2[n-1]/f0 * etax[1] ) )
#                 print S

                    L[n, n-1] = R * ( -D ) - S
                    L[n, n] = R * D - S
                    G[n, n-1] = - D
                    G[n, n] = D     
                    
            #val, func = eigs( lil_matrix(lil_matrix(inv(G)).dot(L)), k=num, which=sort )   # default returns 6 eigenvectors
            val, func = eigs( csc_matrix(csc_matrix(inv(csc_matrix(G))).dot(csc_matrix(L))), 
                             k=num, which=sort )   # default returns 6 eigenvectors

            ###########
            # eigs returns complex values. For a linear-stability analysis, 
            # we don't want the eigenvalues to be real
            ###########
            #tol = 1e-20
            #np.testing.assert_allclose(np.imag(func), 0, atol=tol)
            #np.testing.assert_allclose(np.imag(val), 0, atol=tol)
            #val = np.real(val)
            #func = np.real(func)

            if i == 0 and j == 0:
                omega = np.zeros( (len(val), len(l), len(k)), dtype=complex )
                psi = np.zeros( (nz+1, len(val), len(l), len(k)), dtype=complex )
            omega[:, j, i] = val
            psi[:, :, j, i] = func  # Each column is the eigenfunction
            
    ###########
    # they are often sorted and normalized but not always,
    # so we have to do that here.
    # sort them by the imaginary part of the eigenvalues 
    # (growth rate)
    ###########
    omega1d = omega.reshape( (len(val), len(k)*len(l)) ).imag.max(axis=1)
    p = np.argsort(omega1d)[::-1]
    omega = omega[p]
    psi = psi[:, p]
    
    # return the first four leading modes
    return k, l, zf, omega1d[p], omega[:4], psi[:, :4]
    