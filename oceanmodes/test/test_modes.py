# python 3 forward compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
#
import numpy as np
import unittest
import oceanmodes as modes
#reload(modes)

#############
# Neutral Modes
#############
class TestNeutralModes(unittest.TestCase):
    """class to test Neutral Modes
    """
    def test_topography_truncation(self):
        nz = 10
        z = np.arange(nz)
        f = np.ones(nz)

        zt, ft = modes.baroclinic._maybe_truncate_above_topography(z,f)
        self.assertEqual(len(ft), nz)
        self.assertEqual(len(zt), nz)

        f[-1] = np.nan
        zt, ft = modes.baroclinic._maybe_truncate_above_topography(z,f)
        self.assertEqual(len(ft), nz-1)
        self.assertEqual(len(zt), nz-1)

        f = np.ma.masked_array(f, z==(nz-1))
        zt, ft = modes.baroclinic._maybe_truncate_above_topography(z,f)
        self.assertEqual(len(ft), nz-1)
        self.assertEqual(len(zt), nz-1)

        f.mask[0] = True
        with self.assertRaises(ValueError):
            zt, ft = modes.baroclinic._maybe_truncate_above_topography(z,f)

    def test_depth_kwarg(self):
        nz = 20
        zin = 0.5*nz**-1 + np.arange(nz, dtype=np.float64)/nz
        N2 = np.full(nz, 1.)
        f0 = 1.

        with self.assertRaises(ValueError):
            z, def_radius, bc_modes = modes.neutral_modes_from_N2_profile(
                zin, N2, f0, depth=zin[-5]
            )

        # no error expected
        z, def_radius, bc_modes = modes.neutral_modes_from_N2_profile(
            zin, N2, f0, depth=1.1
        )

    def test_neutral_mode_args(self):
        nz = 10
        N2 = np.zeros(nz)
        depth = np.arange(nz)
        # check for unequal lengths of arrays
        with self.assertRaises(ValueError):
            _, _, _ = modes.neutral_modes_from_N2_profile(
                depth[1:], N2, 1.
            )
        with self.assertRaises(ValueError):
            _, _, _ = modes.neutral_modes_from_N2_profile(
                depth, N2[1:], 1.
            )

        depth_non_monotonic = depth
        depth_non_monotonic[0] = 5
        # check for non-monotonic profile
        with self.assertRaises(ValueError) as cm:
            _, _, _ = modes.neutral_modes_from_N2_profile(
                depth_non_monotonic, N2, 1.
            )

    def test_N2_const_equal_spacing(self):
        # prepare a test N^2 profile
        nz = 20
        depth = 0.5*nz**-1 + np.arange(nz, dtype=np.float64)/nz

        N2 = np.full(nz, 1.)
        f0 = 1.

        z, def_radius, bc_modes = modes.neutral_modes_from_N2_profile(
            depth, N2, f0
        )

        # make sure we got the right number of modes
        # NOPE! don't want all the modes and scipy.sparse.linalg.eigs won't
        # compute them
        #self.assertEqual(nz, len(def_radius))
        # make sure the modes themselves have the right structure
        self.assertEqual(nz+1, bc_modes.shape[0],
            msg='modes array must the right shape')

        self.assertTrue(np.all(np.diff(def_radius[1:]) < 0),
            msg='modes should be sorted in order of decreasing deformation radius')

        nmodes = len(def_radius)
        zero_crossings = np.abs(np.diff(np.sign(bc_modes), axis=0)/2).sum(axis=0)
        self.assertListEqual(list(range(nmodes)), list(zero_crossings),
            msg='modes should have the correct number of zero crossings')

        mode_amplitudes = (bc_modes**2).sum(axis=0)
        self.assertTrue(np.allclose(np.ones(nmodes), mode_amplitudes),
            msg='modes should be normalized to amplitude of 1')
        
        
##########
# Linear Instability
##########
class TestLinearInstability(unittest.TestCase):
    """class to test Linear Instability Modes
    """
    def test_depth_kwarg(self):
        nz = 100
        zin = 0.5*nz**-1 + np.arange(nz, dtype=np.float64)/nz
        N2 = np.full(nz, 1.)
        f0 = 1.
        #beta = 1e-6
        beta = 0.
        Nx = 20
        Ny = 20
        dx = .1
        dy = .1
        ubar = np.zeros((nz+1, 1))
        vbar = np.zeros((nz+1, 1))
        #ubar = 1e-2 * np.ones_like(zin)
        #vbar = 1e-2 * np.ones_like(zin)
        etax = np.zeros((2, 1))
        etay = np.zeros((2, 1))
        #etax = np.array([1e-1*beta, beta])
        #etay = np.array([1e-1*beta, beta])

        with self.assertRaises(ValueError):
            k, l, z, max_growth_rate, growth_rate, vertical_modes = modes.instability_analysis_from_N2_profile(
                zin, N2, f0, beta, Nx, Ny, dx, dy, ubar, vbar, etax, etay, depth=zin[-5]
            )

        # no error expected
        k, l, z, max_growth_rate, growth_rate, vertical_mode = modes.instability_analysis_from_N2_profile(
                zin, N2, f0, beta, Nx, Ny, dx, dy, ubar, vbar, etax, etay, depth=1.1
        )

    def test_linear_instability_args(self):
        nz = 10
        N2 = np.zeros(nz)
        depth = np.arange(nz)
        ubar = 1e-2 * np.ones(nz)
        vbar = 1e-2 * np.ones(nz)
        beta = 1e-6
        etax = np.array([1e-1*beta, beta])
        etay = np.array([1e-1*beta, beta])
        
        ###########
        # check for unequal lengths of arrays
        ###########
        with self.assertRaises(ValueError):
            _, _, _, _, _, _ = modes.instability_analysis_from_N2_profile(
                depth[1:], N2, 1., beta, 20, 20, 1e-1, 1e-1, ubar, vbar, etax, etay
            )
        with self.assertRaises(ValueError):
            _, _, _, _, _, _ = modes.instability_analysis_from_N2_profile(
                depth, N2[1:], 1., beta, 20, 20, 1e-1, 1e-1, ubar, vbar, etax, etay
            )
        #with self.assertRaises(ValueError):
        #    _, _, _, _, _, _, _, _ = modes.instability_analysis_from_N2_profile(
        #        depth, N2, 1., beta, 50, 50, 1e-1, 1e-1, ubar[1:], vbar, etax, etay
        #    )

        depth_non_monotonic = depth
        depth_non_monotonic[0] = 5
        # check for non-monotonic profile
        with self.assertRaises(ValueError) as cm:
            _, _, _, _, _, _ = modes.instability_analysis_from_N2_profile(
                depth_non_monotonic, N2, 1., beta, 50, 50, 1e-1, 1e-1, ubar, vbar, etax, etay
            )

    def test_Eady(self):
        """ Eady setup
        """
        ###########
        # prepare parameters for Eady
        ###########
        nz = 400
        zin = np.arange(nz+1, dtype=np.float64)/nz
        N2 = np.full(nz, 1.)
        f0 = 1.
        beta = 0.
        Nx = int(1e2)
        Ny = int(1e2)
        dx = 1e-1
        dy = 1e-1
        vbar = np.zeros(nz+1)
        ubar = zin
        etax = np.zeros(2)
        etay = np.zeros(2)

        k, l, z, max_growth_rate, growth_rate, vertical_modes = modes.instability_analysis_from_N2_profile(
                .5*(zin[1:]+zin[:-1]), N2, f0, beta, Nx, Ny, dx, dy, ubar, vbar, etax, etay, depth=1., sort='LI', num=2
        )
        
        ###################
        # make sure we got the right number of modes
        # NOPE! don't want all the modes and scipy.sparse.linalg.eigs won't
        # compute them
        # self.assertEqual(nz, len(def_radius))
        # make sure the modes themselves have the right structure
        ###################
        self.assertEqual(nz+1, vertical_modes.shape[0],
            msg='modes array must be in the right shape')

        self.assertTrue(np.all( np.diff( 
                    growth_rate.reshape((growth_rate.shape[0], len(k)*len(l))).imag.max(axis=1) ) <= 0.),
        #self.assertTrue(np.all( max_growth_rate == 0.),
            msg='imaginary part of modes should be descending')
        
        #nmodes = len(def_radius)
        #zero_crossings = np.abs(np.diff(np.sign(bc_modes), axis=0)/2).sum(axis=0)
        #self.assertListEqual(list(range(nmodes)), list(zero_crossings),
        #    msg='modes should have the correct number of zero crossings')

        mode_amplitude1 = (np.absolute(vertical_modes[:, 0])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude1),
            msg='mode1 should be normalized to amplitude of 1 at all horizontal wavenumber points')
        
        mode_amplitude2 = (np.absolute(vertical_modes[:, 1])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude2),
            msg='mode2 should be normalized to amplitude of 1 at all horizontal wavenumber points')
        
        #mode_amplitude3 = (np.absolute(vertical_modes[:, 2])).sum()
        #self.assertTrue(np.allclose(1., mode_amplitude3),
        #    msg='mode3 should be normalized to amplitude of 1')
        
        #########
        # Analytical solution for Eady growth rate
        #########
        growthEady = np.zeros((len(k), 1))
        for i in range(len(k)):
            if (np.tanh(.5*k[i])**-1 - .5*k[i]) * (.5*k[i] - np.tanh(.5*k[i])) < 0:
                growthEady[i] = 0.
            else:
                growthEady[i] = ubar.max() * np.sqrt( (np.tanh(.5*k[i])**-1 - .5*k[i]) * (.5*k[i] - np.tanh(.5*k[i])) )
                
        self.assertTrue( np.allclose(growth_rate.imag[0, 0, :], growthEady, atol=5e-2),
            msg='The numerical growth rates should be close to the analytical Eady solution' )
