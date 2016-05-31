# python 3 forward compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
#
import numpy as np
import unittest
import oceanmodes as modes
from scipy import fftpack as fft
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
        zN2 = 0.5*nz**-1 + np.arange(nz, dtype=np.float64)/nz
        zU = 0.5*nz**-1 + np.arange(nz+1, dtype=np.float64)/nz
        N2 = np.full(nz, 1.)
        f0 = 1.
        #beta = 1e-6
        beta = 0.
        Nx = 1
        Ny = 1
        dx = .1
        dy = .1
        k = fft.fftshift( fft.fftfreq(Nx, dx) )
        l = fft.fftshift( fft.fftfreq(Ny, dy) )
        ubar = np.zeros(nz+1)
        vbar = np.zeros(nz+1)
        etax = np.zeros(2)
        etay = np.zeros(2)

        with self.assertRaises(ValueError):
            z, growth_rate, vertical_modes = modes.instability_analysis_from_N2_profile(
                zN2, N2, f0, beta, k, l, zU, ubar, vbar, etax, etay, depth=zN2[-5]
            )

        # no error expected
        z, growth_rate, vertical_mode = modes.instability_analysis_from_N2_profile(
                zN2, N2, f0, beta, k, l, zU, ubar, vbar, etax, etay, depth=1.1
        )

    def test_linear_instability_args(self):
        nz = 10
        N2 = np.zeros(nz)
        depth = np.arange(nz)
        ubar = 1e-2 * np.ones(nz+1)
        vbar = 1e-2 * np.ones(nz+1)
        beta = 1e-6
        etax = np.array([1e-1*beta, beta])
        etay = np.array([1e-1*beta, beta])
        Nx = 1
        Ny = 1
        dx = .1
        dy = .1
        k = fft.fftshift( fft.fftfreq(Nx, dx) )
        l = fft.fftshift( fft.fftfreq(Ny, dy) )

        ###########
        # check for unequal lengths of arrays
        ###########
        with self.assertRaises(ValueError):
            _, _, _ = modes.instability_analysis_from_N2_profile(
                depth[1:], N2, 1., beta, k, l, depth, ubar, vbar, etax, etay
            )
        with self.assertRaises(ValueError):
            _, _, _ = modes.instability_analysis_from_N2_profile(
                depth, N2[1:], 1., beta, k ,l, depth, ubar, vbar, etax, etay
            )

        depth_non_monotonic = depth
        depth_non_monotonic[0] = 5
        depth_non_monotonic[1] = 5
        # check for non-monotonic profile
        Nx = 50
        Ny = 50
        dx = 1e-1
        dy = 1e-1
        k = fft.fftshift( fft.fftfreq(Nx, dx) )
        l = fft.fftshift( fft.fftfreq(Ny, dy) )
        with self.assertRaises(ValueError) as cm:
            _, _, _ = modes.instability_analysis_from_N2_profile(
                depth_non_monotonic[1:], N2, 1., beta, k, l, depth, ubar, vbar, etax, etay
            )

    def test_Eady(self, atol=5e-2, nz=20, Ah=0.):
        """ Eady setup
        """
        ###########
        # prepare parameters for Eady
        ###########
        nz = nz
        zin = np.arange(nz+1, dtype=np.float64)/nz
        N2 = np.full(nz, 1.)
        f0 = 1.
        beta = 0.
        Nx = 10
        Ny = 1
        dx = 1e-1
        dy = 1e-1
        k = fft.fftshift( fft.fftfreq(Nx, dx) )
        l = fft.fftshift( fft.fftfreq(Ny, dy) )
        vbar = np.zeros(nz+1)
        ubar = zin
        etax = np.zeros(2)
        etay = np.zeros(2)

        z, growth_rate, vertical_modes = \
            modes.instability_analysis_from_N2_profile(
                .5*(zin[1:]+zin[:-1]), N2, f0, beta, k, l,
                zin, ubar, vbar, etax, etay, depth=1., sort='LI', num=2
        )

        self.assertEqual(nz+1, vertical_modes.shape[0],
            msg='modes array must be in the right shape')

        self.assertTrue(np.all( np.diff(
                    growth_rate.reshape((growth_rate.shape[0], len(k)*len(l))).imag.max(axis=1) ) <= 0.),
            msg='imaginary part of modes should be descending')

        mode_amplitude1 = (np.absolute(vertical_modes[:, 0])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude1),
            msg='mode1 should be normalized to amplitude of 1 at all horizontal wavenumber points')

        mode_amplitude2 = (np.absolute(vertical_modes[:, 1])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude2),
            msg='mode2 should be normalized to amplitude of 1 at all horizontal wavenumber points')

        #########
        # Analytical solution for Eady growth rate
        #########
        growthEady = np.zeros(len(k))
        for i in range(len(k)):
            if (k[i]==0) or ((np.tanh(.5*k[i])**-1 - .5*k[i]) * (.5*k[i] - np.tanh(.5*k[i])) < 0):
                growthEady[i] = 0.
            else:
                growthEady[i] = ubar.max() * np.sqrt( (np.tanh(.5*k[i])**-1 - .5*k[i]) * (.5*k[i] - np.tanh(.5*k[i])) )

        self.assertTrue( np.allclose(growth_rate.imag[0, 0, :], growthEady, atol=atol),
            msg='The numerical growth rates should be close to the analytical Eady solution' )
        
    def test_OCCA_GulfStream(self, Ah=0., atol=1e-8):
        """ Actual profiles in the Gulf Stream region 
             (@ 39.5N, 299.5E)
        """
        ###########
        # prepare parameters for Eady
        ###########
        z_u = np.array([5.00000000e+00,1.50000000e+01,2.50000000e+01,3.50000000e+01,
           4.50000000e+01,5.50000000e+01,6.50000000e+01,7.50050049e+01,
           8.50250015e+01,9.50950012e+01,1.05309998e+02,1.15870003e+02,
           1.27150002e+02,1.39739990e+02,1.54470001e+02,1.72399994e+02,
           1.94735001e+02,2.22710007e+02,2.57470001e+02,2.99929993e+02,
           3.50679993e+02,4.09929993e+02,4.77470001e+02,5.52710022e+02,
           6.34735046e+02,7.22400024e+02,8.14470093e+02,9.09740112e+02,
           1.00715503e+03,1.10590503e+03,1.20553503e+03,1.30620508e+03,
           1.40914990e+03,1.51709497e+03,1.63417480e+03,1.76513477e+03,
           1.91414990e+03,2.08403491e+03,2.27622510e+03,2.49125000e+03,
           2.72925000e+03,2.99025000e+03,3.27425000e+03,3.58125000e+03,
           3.91125000e+03,4.26425000e+03,4.64025000e+03,5.03925000e+03,
           5.46125000e+03,5.90625000e+03])
        z_N2 = np.array([10.00006115,20.00006114,30.00006113,40.00006112,50.0000611,
           60.00006109,70.00256358,80.01506451,90.06006328,100.20256308,
          110.5900682,121.51007976,133.44509256,147.10512803,163.43519352,
          183.5678012,208.72298023,240.09073926,278.70109317,325.30655719,
          380.30712264,443.70276088,515.09343642,593.72659767,678.57216844,
          768.44015906,862.11055267,958.45325725,1056.53586028,1155.72595435,
         1255.87608991,1357.68378579,1463.12934261,1575.64299213,1699.66489761,
         1839.65538686,1999.10931291,2180.15155508,2383.76440293,2610.28273438,
         2859.78914881,3132.29607118,3427.80347989,3746.3113517,4087.81966191,
         4452.32838443,4839.83749199,5250.34695635,5683.85674858])
        ubar = np.array([0.23944686,0.202685,0.18200348,0.17409228,0.16885349,0.16435893,
          0.16032954,0.15664397,0.15308236,0.14954343,0.14679012,0.14432698,
          0.1414368,0.13848211,0.13460122,0.13018939,0.1244467,0.11745287,
          0.10890713,0.09883665,0.08717503,0.07444824,0.06123254,0.04839063,
          0.03677421,0.0268672,0.01867473,0.0121845,0.00760518,0.00489767,
          0.00353413,0.00285542,0.0024165,0.00199221,0.00144401,0.0006767,
         -0.00030298,-0.00136913,-0.00236471,-0.00318346,-0.00376874,-0.00404952,
         -0.00395442,-0.00351102,-0.00283759,-0.00198498,-0.00068925,0.00059345,
         np.nan,np.nan])
        vbar = np.array([0.03967041,0.03510151,0.04019022,0.04668061,0.04913285,0.0502381,
          0.05102414,0.0515379,0.05193662,0.05222316,0.05213001,0.05184483,
          0.05137736,0.05068032,0.04974374,0.04858968,0.04708494,0.04507946,
          0.04265563,0.03937409,0.03542616,0.03087777,0.02564656,0.02021027,
          0.0150082,0.01028549,0.00620139,0.00287003,0.00046666,-0.00091146,
         -0.00144855,-0.00152616,-0.00147558,-0.00145375,-0.00149181,-0.00157445,
         -0.00168814,-0.0018392,-0.00204265,-0.00229426,-0.00256875,-0.0028379,
         -0.00305751,-0.0031456,-0.00304893,-0.00274524,-0.00244838,np.nan,
         np.nan,np.nan])
        Rd = 25787.38588424
        N2 = np.array([4.49013733e-05,1.32619531e-04,1.67213349e-04,1.39980381e-04,
           1.08620176e-04,8.89353316e-05,7.59696940e-05,6.85660114e-05,
           6.13053056e-05,4.96623307e-05,4.26153730e-05,3.83933836e-05,
           3.33244718e-05,3.14573673e-05,2.74780120e-05,2.50869570e-05,
           2.35907064e-05,2.26047540e-05,2.06309696e-05,1.93499569e-05,
           1.85802260e-05,1.73887368e-05,1.49995125e-05,1.25945779e-05,
           1.06210281e-05,9.69791903e-06,9.18845407e-06,7.56655382e-06,
           5.00383224e-06,2.83633215e-06,1.67170878e-06,1.19350982e-06,
           9.87252978e-07,9.54096555e-07,1.05979705e-06,1.18957478e-06,
           1.23432375e-06,1.22190029e-06,1.20084648e-06,1.18287498e-06,
           1.14801849e-06,1.08643614e-06,9.36511867e-07,6.17342678e-07,
           3.35507108e-07,3.19646379e-07,2.60733349e-07,np.nan,
            np.nan])
        f0 = 9.27671062527e-05
        beta = 1.76637107718e-11
        Nx = 100; Ny = 1
        dx = 1e3; dy = 1e3
        k = fft.fftshift( fft.fftfreq(Nx, dx) )
        l = fft.fftshift( fft.fftfreq(Ny, dy) )
        k = k[np.absolute(k) < 5.*Rd**-1]
        etax = np.zeros(2)
        etay = np.zeros(2)

        z, growth_rate, vertical_modes = \
            modes.instability_analysis_from_N2_profile(
                z_N2, N2, f0, beta, k, l,
                z_u, ubar, vbar, etax, etay, sort='LI', num=2
        )
        
        nz = len(z)

        self.assertEqual(nz, vertical_modes.shape[0],
            msg='modes array must be in the right shape')

        self.assertTrue(np.all( np.diff(
                    growth_rate.reshape((growth_rate.shape[0], len(k)*len(l))).imag.max(axis=1) ) <= 0.),
            msg='imaginary part of modes should be descending')

        mode_amplitude1 = (np.absolute(vertical_modes[:, 0])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude1),
            msg='mode1 should be normalized to amplitude of 1 at all horizontal wavenumber points')

        mode_amplitude2 = (np.absolute(vertical_modes[:, 1])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude2),
            msg='mode2 should be normalized to amplitude of 1 at all horizontal wavenumber points')
        
        ##########
        # Row corresponding to l=0 of growth mode
        ##########
        w_sol = np.array([1.09940514e-06,9.63383313e-07,8.32358326e-07,6.99704577e-07,
           5.60854037e-07,4.16059352e-07,2.69400582e-07,1.57391758e-07,
           1.57022966e-07,1.39924035e-07,1.00847043e-07,5.20661606e-08,
           1.14405052e-07,1.90974306e-07,1.83088454e-07,3.63353900e-07,
           5.49322650e-07,2.90420152e-07,1.50720341e-08,0.00000000e+00,
           1.50720341e-08,2.90420152e-07,5.49322650e-07,3.63353900e-07,
           1.83088454e-07,1.90974306e-07,1.14405052e-07,5.20661606e-08,
           1.00847043e-07,1.39924035e-07,1.57022966e-07,1.57391758e-07,
           2.69400582e-07,4.16059352e-07,5.60854037e-07,6.99704577e-07,
           8.32358326e-07,9.63383313e-07,1.09940514e-06])
        
        self.assertTrue( np.allclose(growth_rate.imag[0, 0], w_sol, atol=atol),
            msg='The OCCA numerical growth rates should be close to the solution' )