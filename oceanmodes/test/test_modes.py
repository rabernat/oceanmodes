# python 3 forward compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
#
import numpy as np
import unittest
import oceanmodes as modes

class TestNeutralModes(unittest.TestCase):

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
