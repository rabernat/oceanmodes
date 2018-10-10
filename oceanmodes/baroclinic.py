# python 3 forward compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
#
import sys
import numpy as np
from scipy import fftpack as fft
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import eigs, inv
import warnings

def _maybe_truncate_above_topography(z, *args):
    """Return truncated versions of z and arg if arg is masked or nan.
    """
    ndata = len(args)
    
    # checks on shapes of stuff
    for n in range(ndata):
        if not z.shape == args[n].shape:
            raise ValueError('z and f must have the same length')
            
        if ndata > 1 and n < ndata-1:
            if not args[n].shape == args[n+1].shape:
                raise ValueError('All variables should have same length before truncation')
            if not (np.ma.masked_invalid(args[n]).mask.all() 
                    == np.ma.masked_invalid(args[n+1]).mask.all()):
                raise ValueError('All variables must have same mask before truncation')
            if not (np.ma.masked_invalid(args[n]).compressed().shape 
                    == np.ma.masked_invalid(args[n+1]).compressed().shape):
                raise ValueError('All variables must have same length after truncation')

        fm = np.ma.masked_invalid(args[n])
        
        # check to make sure the mask is only at the bottom
        # if not fm.mask[-1]:
        #     raise ValueError('topography should be at the bottom of the column')
        # the above check was redundant with this one
        if fm.mask[-1] and np.diff(fm.mask).sum() != 1:
            raise ValueError('topographic mask should be monotonic')
        
        if ndata > 1:    
            if n == 0:
                fout = np.zeros((ndata, len(fm.compressed())))
            fout[n] = fm.compressed()
        else:
            fout = fm.compressed()
   
    zout = z[~fm.mask]
    
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
    if z[0] == 0:
        raise ValueError('z[0] should be non-zero, as it is a point where gradient quantity (N2) is defined')
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

    # this gets the eigenvalues and eigenvectors
    if nz <= 2:
        w, v = eigs(L, k=nz-1, which='SM')
    elif nz > 2 and len(kwargs) > 0:
        w, v = eigs( L, k=kwargs['num_eigen'], which='SM', v0=kwargs['init_vector'], 
                        ncv=kwargs['num_Lanczos'], maxiter=kwargs['iteration'],
                        tol=kwargs['tolerance'])
    else:
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
    Rd = (-w)**-0.5 / np.absolute(f0)

    return zf, Rd, v

def instability_analysis_from_N2_profile(zN2, N2, f0, beta, k, l, zU, ubar, vbar, etax, etay, Ah=0.,
                                         sort='LI', num=4, depth=None, **kwargs):
    """Calculate baroclinic unstable modes from a profile of buoyancy frequency.
        Solves the eigenvalue problem

            \frac{\partial q}{\partial t} + \boldsymbol{U} \cdot \nabla q + \boldsymbol{u} \cdot \nabla Q = Ah \nabla^2 q \ \ \ \ \ (1)
            
        with the boundary conditions of

            \frac{\partial b}{\partial t} + \boldsymbol{U} \cdot \nabla b + \boldsymbol{u} \cdot \nabla (B + N^2 \eta) = 0 \ \ \ \ \ (2)
            
        at the top and bottom. Each variable is defined as

            q = \nabla^2 \ \psi + \Gamma \ \psi \ , \ \ \ \ Q = \beta y + \Gamma \ \Psi  \ \ \ \ \ (3)

        where $\Gamma := \frac{\partial}{\partial z} \Big( \frac{f^2}{N^2} \frac{\partial}{\partial z} \Big)$ and

            b = f \frac{\partial \psi}{\partial z} \ , \ \ \ \ B = f \frac{\partial \Psi}{\partial z} \ , \ \ \ \ N^2 = - \frac{g}{\rho_0} \frac{\partial \overline{\rho}}{\partial z} \ \ \ \ \ (4)

            \frac{\partial}{\partial z} ( \boldsymbol{f} \times \boldsymbol{u} ) = \nabla b = \frac{g}{\rho_0} \nabla \overline{\rho} \ \ \ \ \ (5)
            
        Assuming a plane-wave solution, viz.

            \psi = Re \big[ \hat{\psi}(z) e^{i(kx + ly - \omega t)} \big]
        
        we get

            \nabla^2 \psi = - (k^2 + l^2) \ \psi = -K^2 \psi \ , \ \ \ u = - \frac{\partial \psi}{\partial y} = -il \ \psi \ , \ \ \ v = \frac{\partial \psi}{\partial x} = ik \ \psi

            \frac{\partial}{\partial x} q = ik (-K^2 + \Gamma) \ \psi \ , \ \ \ \frac{\partial}{\partial y} q = il (-K^2 + \Gamma) \ \psi

            \frac{\partial}{\partial x} Q = \Gamma V \ , \ \ \ \frac{\partial}{\partial y} Q = \beta - \Gamma U

            \frac{\partial}{\partial x} b = ikf \frac{\partial \psi}{\partial z} \ , \ \ \ \frac{\partial}{\partial y} b = ilf \frac{\partial \psi}{\partial z}

            \frac{\partial}{\partial x} B = f \frac{\partial V}{\partial z} \ , \ \ \ \frac{\partial}{\partial y} B = - f \frac{\partial U}{\partial z}
            
        so eqn. (1) becomes

            -i \omega (-K^2 + \Gamma) \ \hat{\psi} + ik U (-K^2 + \Gamma) \ \hat{\psi} + il V (-K^2 + \Gamma) \ \hat{\psi} - il \frac{\partial Q}{\partial x} \ \hat{\psi} + ik \frac{\partial Q}{\partial y} \ \hat{\psi} = - Ah K^2 \hat{\psi}

            \therefore \ \ (\boldsymbol{K} \cdot \boldsymbol{U} - \omega - i Ah K^2) (\Gamma - K^2) \ \hat{\psi} = - \bigg( k \frac{\partial Q}{\partial y} - l \frac{\partial Q}{\partial x} \bigg) \ \hat{\psi} \ \ \ \ \ (6)
        
        and eqn. (2) becomes

            -i \omega f \frac{\partial \hat{\psi}}{\partial z} + ikf U \frac{\partial \hat{\psi}}{\partial z} + ilf V \frac{\partial \hat{\psi}}{\partial z} - il \bigg( f \frac{\partial V}{\partial z} + N^2 \frac{\partial \eta}{\partial x} \bigg) \ \hat{\psi} + ik \bigg( -f \frac{\partial U}{\partial z} + N^2 \frac{\partial \eta}{\partial y} \bigg) \ \hat{\psi} = 0

            \therefore \ \ (\boldsymbol{K} \cdot \boldsymbol{U} - \omega) \frac{\partial \hat{\psi}}{\partial z} = \bigg[ k \bigg( \frac{\partial U}{\partial z} - \frac{N^2}{f} \frac{\partial \eta}{\partial y} \bigg) + l \bigg( \frac{\partial V}{\partial z} + \frac{N^2}{f} \frac{\partial \eta}{\partial x} \bigg) \bigg] \ \hat{\psi} \ \ \ \ \ (7)
            
        Now the eigenfunctions are \hat{\psi} and eigenvalues are \omega so rewriting eqn. (6) and (7) gives

            \bigg[ ( kU + lV - i Ah K^2 ) \bigg( \frac{\partial}{\partial z} \frac{f^2}{N^2} \frac{\partial}{\partial z} - K^2 \bigg) + \bigg( k \frac{\partial Q}{\partial y} - l \frac{\partial Q}{\partial x} \bigg) \bigg] \ \hat{\psi} = \omega \bigg( \frac{\partial}{\partial z} \frac{f^2}{N^2} \frac{\partial}{\partial z} - K^2 \bigg) \ \hat{\psi} \ \ \ \ \ (8)

            \bigg[ ( kU + lV ) \frac{\partial}{\partial z} - k \bigg( \frac{\partial U}{\partial z} - \frac{N^2}{f} \frac{\partial \eta}{\partial y} \bigg) - l \bigg( \frac{\partial V}{\partial z} + \frac{N^2}{f} \frac{\partial \eta}{\partial x} \bigg) \bigg] \ \hat{\psi} = \omega \frac{\partial}{\partial z} \ \hat{\psi} \ \ \ \ \ (9)
            
        Hence,

            \boldsymbol{B}^{-1} \boldsymbol{A} \ \hat{\psi} = \omega \ \hat{\psi} \ \ \ \ (-H < z < 0)

            \boldsymbol{D}^{-1} \boldsymbol{C} \ \hat{\psi} = \omega \ \hat{\psi} \ \ \ \ \ (z = 0, -H)

        where

            \boldsymbol{A} = (kU + lV - i Ah K^2) \bigg( \frac{\partial}{\partial z} \frac{f^2}{N^2} \frac{\partial}{\partial z} - K^2 \bigg) + \bigg( k \frac{\partial Q}{\partial y} - l \frac{\partial Q}{\partial x} \bigg)

            \boldsymbol{B} = \frac{\partial}{\partial z} \frac{f^2}{N^2} \frac{\partial}{\partial z} - K^2

            \boldsymbol{C} = (kU + lV) \frac{\partial}{\partial z} - k \bigg( \frac{\partial U}{\partial z} - \frac{N^2}{f} \frac{\partial \eta}{\partial y} \bigg) - l \bigg( \frac{\partial V}{\partial z} + \frac{N^2}{f} \frac{\partial \eta}{\partial x} \bigg)

            \boldsymbol{D} = \frac{\partial}{\partial z}
            
        Discretizing the second derivative term gives

            \frac{\partial}{\partial z} \frac{f^2}{N^2} \frac{\partial}{\partial z} \ \hat{\psi} = \frac{\partial}{\partial z} \bigg( \frac{f^2}{N^2} \bigg) \frac{\partial \hat{\psi}}{\partial z} + \frac{f^2}{N^2} \frac{\partial^2 \hat{\psi}}{\partial z^2} \\ = \frac{f^2}{\delta_n} \bigg( \frac{1}{N_{n-1}^2} - \frac{1}{N_n^2} \bigg) \frac{1}{2} \bigg( \frac{\partial \hat{\psi}}{\partial z} \bigg|_n + \frac{\partial \hat{\psi}}{\partial z} \bigg|_{n-1} \bigg) + \frac{f^2}{2} \bigg( \frac{1}{N_n^2} + \frac{1}{N_{n-1}^2} \bigg) \frac{1}{\delta_n} \bigg( \frac{\partial \hat{\psi}}{\partial z} \bigg|_{n-1} - \frac{\partial \hat{\psi}}{\partial z} \bigg|_n \bigg) \\ = \frac{f^2}{\delta_n} \bigg( \frac{1}{N_{n-1}^2} - \frac{1}{N_n^2} \bigg) \frac{1}{2} \bigg( \frac{\hat{\psi}_n - \hat{\psi}_{n+1}}{\Delta_n} + \frac{\hat{\psi}_{n-1} - \hat{\psi}_n}{\Delta_{n-1}} \bigg) + \frac{f^2}{2} \bigg( \frac{1}{N_n^2} + \frac{1}{N_{n-1}^2} \bigg) \frac{1}{\delta_n} \bigg( \frac{\hat{\psi}_{n-1} - \hat{\psi}_n}{\Delta_{n-1}} - \frac{\hat{\psi}_n - \hat{\psi}_{n+1}}{\Delta_n} \bigg) \\ = \frac{f^2}{\delta_n} \bigg[ \frac{\hat{\psi}_{n+1}}{N_n^2 \Delta_n} - \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) \hat{\psi}_n + \frac{\hat{\psi}_{n-1}}{N_{n-1}^2 \Delta_{n-1}} \bigg]
            
        so for the interior ocean (0 < z < H), eqn. (8) becomes

            (kU_n + lV_n - i Ah K^2) \bigg[ \frac{f^2}{\delta_n} \bigg\{ \frac{\hat{\psi}_{n+1}}{N_n^2 \Delta_n} - \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) \hat{\psi}_n + \frac{\hat{\psi}_{n-1}}{N_{n-1}^2 \Delta_{n-1}} \bigg\} - K^2 \hat{\psi}_n \bigg] + \bigg( k \frac{\partial Q}{\partial y} \bigg|_n - l \frac{\partial Q}{\partial x} \bigg|_n \bigg) \hat{\psi}_n = \omega \bigg[ \frac{f^2}{\delta_n} \bigg\{ \frac{\hat{\psi}_{n+1}}{N_n^2 \Delta_n} - \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) \hat{\psi}_n + \frac{\hat{\psi}_{n-1}}{N^2_{n-1} \Delta_{n-1}} \bigg\} - K^2 \hat{\psi}_n \bigg]
            
        Thus, for \hat{\psi}_{n-1} :

            (kU_n + lV_n - i Ah K^2) \frac{f^2}{\delta_n} \frac{1}{N_{n-1}^2 \Delta_{n-1}} \hat{\psi}_{n-1} = \omega \frac{f^2}{\delta_n} \frac{1}{N_{n-1}^2 \Delta_{n-1}} \hat{\psi}_{n-1}

        and \hat{\psi}_n :

            (kU_n + lV_n - i Ah K^2) \bigg\{ - \frac{f^2}{\delta_n} \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) - K^2 \bigg\} + k \frac{\partial Q}{\partial y} \bigg|_n - l \frac{\partial Q}{\partial x} \bigg|_n \bigg] \hat{\psi}_n = \omega \bigg\{ - \frac{f^2}{\delta_n} \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) - K^2 \bigg\} \hat{\psi}_n

        and \hat{\psi}_{n+1} :

            (kU_n + lV_n - i Ah K^2) \frac{f^2}{\delta_n} \frac{1}{N_n^2 \Delta_n} \hat{\psi}_{n+1} = \omega \frac{f^2}{\delta_n} \frac{1}{N_n^2 \Delta_n} \hat{\psi}_{n+1}
            
        where

            \frac{\partial Q}{\partial x} \bigg|_n = \frac{f^2}{\delta_n} \bigg\{ \frac{V_{n+1}}{N_n^2 \Delta_n} - \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) V_n + \frac{V_{n-1}}{N_{n-1}^2 \Delta_{n-1}} \bigg\}

            \frac{\partial Q}{\partial y} \bigg|_n = \beta - \frac{f^2}{\delta_n} \bigg\{ \frac{U_{n+1}}{N_n^2 \Delta_n} - \bigg( \frac{1}{N_n^2 \Delta_n} + \frac{1}{N_{n-1}^2 \Delta_{n-1}} \bigg) U_n + \frac{U_{n-1}}{N_{n-1}^2 \Delta_{n-1}} \bigg\}
            
        At the boundaries, eqn. (9) becomes

            \bigg( k \frac{U_0 + U_1}{2} + l \frac{V_0 + V_1}{2} \bigg) \frac{\hat{\psi}_0 - \hat{\psi}_1}{\Delta_0} - \bigg\{ k \bigg( \frac{U_0 - U_1}{\Delta_0} - \frac{N_0^2}{f} \frac{\partial \eta_s}{\partial y} \bigg) + l \bigg( \frac{V_0 - V_1}{\Delta_0} + \frac{N_0^2}{f} \frac{\partial \eta_s}{\partial x} \bigg) \bigg\} \frac{\hat{\psi}_1 + \hat{\psi}_0}{2} = \omega \frac{\hat{\psi}_0 - \hat{\psi}_1}{\Delta_0}

        Therefore at the surface \psi_0 :

            \bigg[ \bigg( k \frac{U_0 + U_1}{2} + l \frac{V_0 + V_1}{2} \bigg) \frac{1}{\Delta_0} - \frac{1}{2} \bigg\{ k \bigg( \frac{U_0 - U_1}{\Delta_0} - \frac{N_0^2}{f} \frac{\partial \eta_s}{\partial y} \bigg) + l \bigg( \frac{V_0 - V_1}{\Delta_0} + \frac{N_0^2}{f} \frac{\partial \eta_s}{\partial x} \bigg) \bigg\} \bigg] \ \hat{\psi}_0 = \omega \frac{1}{\Delta_0} \hat{\psi}_0
            
        and \psi_1 :

            \bigg[ \bigg( k \frac{U_0 + U_1}{2} + l \frac{V_0 + V_1}{2} \bigg) \frac{-1}{\Delta_0} - \frac{1}{2} \bigg\{ k \bigg( \frac{U_0 - U_1}{\Delta_0} - \frac{N_0^2}{f} \frac{\partial \eta_s}{\partial y} \bigg) + l \bigg( \frac{V_0 - V_1}{\Delta_0} + \frac{N_0^2}{f} \frac{\partial \eta_s}{\partial x} \bigg) \bigg\} \bigg] \ \hat{\psi}_1 = \omega \frac{-1}{\Delta_0} \hat{\psi}_1

        and the same applies for the bottom \psi_{J-1} :

            \bigg[ \bigg( k \frac{U_{J-1} + U_J}{2} + l \frac{V_{J-1} + V_J}{2} \bigg) \frac{1}{\Delta_{J-1}} - \frac{1}{2} \bigg\{ k \bigg( \frac{U_{J-1} - U_J}{\Delta_{J-1}} - \frac{N_{J-1}^2}{f} \frac{\partial \eta_b}{\partial y} \bigg) + l \bigg( \frac{V_{J-1} - V_J}{\Delta_{J-1}} + \frac{N_{J-1}^2}{f} \frac{\partial \eta_b}{\partial x} \bigg) \bigg\} \bigg] \ \hat{\psi}_{J-1} = \omega \frac{1}{\Delta_{J-1}} \hat{\psi}_{J-1}

        and \psi_J :

            \bigg[ \bigg( k \frac{U_{J-1} + U_J}{2} + l \frac{V_{J-1} + V_J}{2} \bigg) \frac{-1}{\Delta_{J-1}} - \frac{1}{2} \bigg\{ k \bigg( \frac{U_{J-1} - U_J}{\Delta_{J-1}} - \frac{N_{J-1}^2}{f} \frac{\partial \eta_b}{\partial y} \bigg) + l \bigg( \frac{V_{J-1} - V_J}{\Delta_{J-1}} + \frac{N_{J-1}^2}{f} \frac{\partial \eta_b}{\partial x} \bigg) \bigg\} \bigg] \ \hat{\psi}_J = \omega \frac{-1}{\Delta_{J-1}} \hat{\psi}_J
    
    
        Parameters
        ----------
        zN2 : array_like
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
        k, l : float
            Parameters to define the horizontal wavenumber
        zU : array_like
            The depths at which ubar/vbar are given. Starts shallow, increases
            positive downward.
        ubar, vbar : array_like
            Time averaged horizontal velocitites [units m/s]. Points below topography
            should be masked or nan.
        Ah : float
            Lateral viscosity
        sort: string, optional
            Sorts the eigenvalues depending on the given argument
        num: integer, optional
            Number of eigenvalues returned
        depth : float, optional
            Total water column depth. If missing will be inferred from z.
        kwargs : optional
            Additional parameters to pass to scipy.sparse.linalg.eigs for
            eigenvalue computation (check documentation:
            http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.sparse.linalg.eigs.html
            for how to define other parameters)
    
    
        Returns
        -------
        zpsi : array_like
            The depths at which phi is defined. Different from z.
        psi : array_like
            eigenvectors
        omega: array_like
            eigenvalues (imaginary part is the growth rate)
    """
    nz_orig = len(zN2)
    zc, N2 = _maybe_truncate_above_topography(zN2, N2)
    dzc = np.hstack(np.diff(zc))
    zf, UV = _maybe_truncate_above_topography(zU, ubar, vbar)
    dzf = np.hstack(np.diff(zf))
    ubar = UV[0]
    vbar = UV[1]
    # make sure length of N2 is one shorter than velocities
    if len(N2) > len(ubar)-1:
        raise ValueError('N2 has the same or longer length than horizontal velocities')
    elif len(N2) < len(ubar)-1:
        warnings.warn('N2 is shorter than horizontal velocities by more than one element')
    
    # make sure z is increasing
    if not np.all(dzc > 0):
        raise ValueError('z should be monotonically increasing')
    if not np.all(dzf > 0):
        raise ValueError('z should be monotonically increasing')
    if depth is None:
        depth = zf[-1]
    else:
        if depth < zf[-1]:
            raise ValueError('depth should not be less than maximum z')
    
    return _instability_analysis_from_N2_profile_raw(
                                    zc, N2, f0, beta, k, l, zf, ubar, vbar, etax, etay, Ah,
                                    sort=sort, num=num, depth=depth, **kwargs)

def _instability_analysis_from_N2_profile_raw(zc, N2, f0, beta, k, l, zf, ubar, vbar, etax, etay, Ah,
                                              sort='LI', num=4, depth=None, **kwargs):
    nz = len(zc)

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
    dzc = np.hstack(np.diff(zc))
    dzf = np.diff(zf)
    
    ##################
    # We want a matrix representation of the operator such that
    #
    #     np.dot(L, psi) = omega * np.dot(G, psi)
    ##################
    
    omega = np.zeros( (num, len(l), len(k)), dtype=np.complex128 )
    psi = np.zeros( (nz+1, num, len(l), len(k)), dtype=np.complex128 )
    
    for j in range(len(l)):
        for i in range(len(k)):
            
            if Ah == 0.:
                L = lil_matrix((nz+1, nz+1), dtype=np.float64)
            else:
                L = lil_matrix((nz+1, nz+1), dtype=np.complex128)
            G = L.copy()

            ################
            # n = 0 (surface)
            #
            # From the discretized equations above, at the surface we get
            #     L[0, 0] = (( k (U[0] + U[1])/2 + l (V[0] + V[1])/2 ) / dzf[0] 
            #                 - .5*( k ( (U[0] - U[1])/dzf[0] - N2[0]**2/f0 * etay[0] ) + l ( (V[0] - V[1])/dzf[0] + N2[0]**2/f0 * etax[0] ))) 
            #     L[0, 1] = (( k (U[0] + U[1])/2 + l (V[0] + V[1])/2 ) / (-dzf[0]) 
            #                 - .5*( k ( (U[0] - U[1])/dzf[0] - N2[0]/f0 * etay[0] ) + l ( (V[0] - V[1])/dzf[0] + N2[0]/f0 * etax[0] )))
            # and
            #     G[0, 0] = dzf[0]**-1
            #     G[0, 1] = -dzf[0]**-1
            ################
            R = k[i] * .5*(ubar[0]+ubar[1]) + l[j] * .5*(vbar[0]+vbar[1]) 
            D = dzf[0]**-1
            S = .5 * ( k[i] * ( (ubar[0]-ubar[1])*D - N2[0]/f0 * etay[0] )
                                                + l[j] * ( (vbar[0]-vbar[1])*D + N2[0]/f0 * etax[0] ) )

            L[0, 0] = R * D - S
            L[0, 1] = R * (-D) - S
            G[0, 0] = D
            G[0, 1] = - D
            # L[0, 0] = R - S * D**-1
            # L[0, 1] = R * (-1.) - S * D**-1
            # G[0, 0] = 1.
            # G[0, 1] = - 1.

            ################
            # n = nz (bottom)
            #
            # From symmetry, at the bottom
            #     L[nz, nz-1] = (( k (U[nz-1] + U[nz])/2 + l (V[nz-1] + V[nz])/2 ) / dzf[nz-1] 
            #           - .5*( k ( (U[nz-1] - U[nz])/dzf[nz-1] - N2[nz-1]**2/f0 * etay[1] ) + l ( (V[nz-1] - V[nz])/dzf[nz-1] + N2[nz-1]**2/f0 * etax[1] ))) 
            #     L[nz, nz] = (( k (U[nz-1] + U[nz])/2 + l (V[nz-1] + V[nz])/2 ) / (-dzf[nz-1]) 
            #           - .5*( k ( (U[nz-1] - U[nz])/dzf[nz-1] - N2[nz-1]/f0 * etay[1] ) + l ( (V[nz-1] - V[nz])/dzf[nz-1] + N2[nz-1]/f0 * etax[1] )))
            # and
            #     G[nz, nz-1] = dzf[nz-1]**-1
            #     G[nz, nz] = -dzf[nz-1]**-1
            ################
            R = k[i] * .5*(ubar[nz-1]+ubar[nz]) + l[j] * .5*(vbar[nz-1]+vbar[nz]) 
            D = dzf[nz-1]**-1
            S = .5 * ( k[i] * ( (ubar[nz-1]-ubar[nz])*D - N2[nz-1]/f0 * etay[1] )
                                                + l[j] * ( (vbar[nz-1]-vbar[nz])*D + N2[nz-1]/f0 * etax[1] ) )

            L[nz, nz-1] = R * D - S
            L[nz, nz] = R * (-D) - S
            G[nz, nz-1] = D
            G[nz, nz] = - D
            # L[nz, nz-1] = R - S * D**-1
            # L[nz, nz] = R * (-1.) - S * D**-1
            # G[nz, nz-1] = 1.
            # G[nz, nz] = - 1.

            ################
            # 0 < n < nz (interior)
            #
            # In the interior, we have
            #     L[n, n-1] = (k U[n] + l V[n] - i Ah K**2) * f**2/dzc[n] / (N2[n-1] * dzf[n-1])
            #     L[n, n] = (k U[n] + l V[n] - i Ah K**2) 
            #                     * ( - f**2/dzc[n] * ( 1/(N2[n] * dzf[n]) + 1/(N2[n-1]*dzf[n-1]) ) - K**2 ) + k*Qy[n] - l*Qx[n] )
            #     L[n, n+1] = (k U[n] + l V[n] - i Ah K**2) * f**2/dzc[n] / (N2[n] * dzf[n])
            # and
            #     G[n, n-1] = f**2/dzc[n] / N2[n-1] / dzf[n-1]
            #     G[n, n] = ( - f**2/dzc[n] * ( 1/(N2[n]*dzf[n]) + 1/(N2[n-1]*dzf[n-1]) ) - K**2 )
            #     G[n, n+1] = f**2/dzc[n] / N2[n] / dzf[n]
            ################
            for n in range(1,nz):
                
                K2 = k[i]**2 + l[j]**2
                if Ah == 0.:
                    R = k[i] * ubar[n] + l[j] * vbar[n]
                else:
                    R = k[i] * ubar[n] + l[j] * vbar[n] - 1j * Ah * K2
                bf = f0**2 * dzc[n-1]**-1
                b_1 = N2[n-1] * dzf[n-1]
                b = N2[n] * dzf[n]
                B_1 = bf * b_1**-1 
                B = - (bf * (b**-1 + b_1**-1) + K2)
                Bt1 = bf * b**-1 

                N2Z = (N2[n]*dzf[n])**-1
                N2Z_1 = (N2[n-1]*dzf[n-1])**-1
                P = ( k[i] * ( beta - bf * ( ubar[n+1] * N2Z
                                                                - (N2Z + N2Z_1) * ubar[n]
                                                                + N2Z_1 * ubar[n-1] ) )
                            - l[j] * bf * ( vbar[n+1] * N2Z 
                                                           - (N2Z + N2Z_1) * vbar[n]
                                                           + N2Z_1 * vbar[n-1] )
                            )

                L[n, n-1] = R * B_1
                L[n, n] = R * B + P
                L[n, n+1] = R * Bt1
                G[n, n-1] = B_1
                G[n, n] = B
                G[n, n+1] = Bt1
        #             L[n, n-1] = R
        #             L[n, n] = R + P * B**-1
        #             L[n, n+1] = R
        #             G[n, n-1] = 1.
        #             G[n, n] = 1.
        #             G[n, n+1] = 1.        
                    
            # Return Nans if matrix G is singular
            try:
                # read in kwargs if any
                if len(kwargs) > 0:
                    val, func = eigs( csc_matrix(inv(csc_matrix(G)).dot(csc_matrix(L))), 
                                                 k=num, which='LI', v0=kwargs['init_vector'], 
                                     ncv=kwargs['num_Lanczos'], maxiter=kwargs['iteration'],
                                    tol=kwargs['tolerance'])  # default returns 6 eigenvectors
                else:
                    num_Lanczos = nz
                    iteration = 10*nz
                    val, func = eigs( csc_matrix(inv(csc_matrix(G)).dot(csc_matrix(L))), 
                                                 k=num, which='LI', ncv=num_Lanczos, maxiter=iteration,
                                       )  # default returns 6 eigenvectors
                # val, func = eigs( csc_matrix(L), M=csc_matrix(G), Minv=csc_matrix(inv(csc_matrix(G))), 
                #                 k=num, which='LI', ncv=num_Lanczos, maxiter=iteration, **kwargs )

                ###########
                # eigs returns complex values. For a linear-stability analysis, 
                # we don't want the eigenvalues to be real
                ###########
                omega[:, j, i] = val
                psi[:, :, j, i] = func  # Each column is the eigenfunction
            
            except RuntimeError:
                warnings.warn('The matrix is ill-conditioned or singular', RuntimeWarning)
                omega[:, j, i] = np.nan
                psi[:, :, j, i] = np.nan
    
    ###########
    # they are often sorted and normalized but not always,
    # so we have to do that here.
    # sort them by the imaginary part of the eigenvalues 
    # (growth rate)
    ###########
    omega1d = omega.reshape( (num, len(k)*len(l)) ).imag.max(axis=1)
    p = np.argsort(omega1d)[::-1]
    omega = omega[p]
    psi = psi[:, p]
    
    # return the first 'num' leading modes
    return zf, omega[:num], psi[:, :num]
    
