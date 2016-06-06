# python 3 forward compatibility
from __future__ import absolute_import, division, print_function
from builtins import *
#
import numpy as np
import unittest
import warnings
import oceanmodes as modes
from scipy import fftpack as fft
# reload(modes)

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
        
        g = np.ones(nz+1)
        with self.assertRaises(ValueError):
            zt, ft, gt = modes.baroclinic._maybe_truncate_above_topography(z,f,g)
        
        zg = np.arange(nz+1)
        g = np.ma.masked_array(g, zg==(nz-1))
        g.mask[-2] = True
        with self.assertRaises(ValueError):
            zt, ft, gt = modes.baroclinic._maybe_truncate_above_topography(z,f,g[:-1])

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
        
    def test_neutral_mode_OCCA_GulfStream(self):
        """ Test profile in the Gulf Stream in the OCCA data set
        """
        zN2 = np.array([  -10.00006115,   -20.00006114,   -30.00006113,   -40.00006112,
         -50.0000611 ,   -60.00006109,   -70.00256358,   -80.01506451,
         -90.06006328,  -100.20256308,  -110.5900682 ,  -121.51007976,
        -133.44509256,  -147.10512803,  -163.43519352,  -183.5678012 ,
        -208.72298023,  -240.09073926,  -278.70109317,  -325.30655719,
        -380.30712264,  -443.70276088,  -515.09343642,  -593.72659767,
        -678.57216844,  -768.44015906,  -862.11055267,  -958.45325725,
       -1056.53586028, -1155.72595435, -1255.87608991, -1357.68378579,
       -1463.12934261, -1575.64299213, -1699.66489761, -1839.65538686,
       -1999.10931291, -2180.15155508, -2383.76440293, -2610.28273438,
       -2859.78914881, -3132.29607118, -3427.80347989, -3746.3113517 ,
       -4087.81966191, -4452.32838443, -4839.83749199, -5250.34695635,
       -5683.85674858])
        N2 = np.array([  4.49013733e-05,   1.32619531e-04,   1.67213349e-04,
         1.39980381e-04,   1.08620176e-04,   8.89353316e-05,
         7.59696940e-05,   6.85660114e-05,   6.13053056e-05,
         4.96623307e-05,   4.26153730e-05,   3.83933836e-05,
         3.33244718e-05,   3.14573673e-05,   2.74780120e-05,
         2.50869570e-05,   2.35907064e-05,   2.26047540e-05,
         2.06309696e-05,   1.93499569e-05,   1.85802260e-05,
         1.73887368e-05,   1.49995125e-05,   1.25945779e-05,
         1.06210281e-05,   9.69791903e-06,   9.18845407e-06,
         7.56655382e-06,   5.00383224e-06,   2.83633215e-06,
         1.67170878e-06,   1.19350982e-06,   9.87252978e-07,
         9.54096555e-07,   1.05979705e-06,   1.18957478e-06,
         1.23432375e-06,   1.22190029e-06,   1.20084648e-06,
         1.18287498e-06,   1.14801849e-06,   1.08643614e-06,
         9.36511867e-07,   6.17342678e-07,   3.35507108e-07,
         3.19646379e-07,   2.60733349e-07,              np.nan,
                    np.nan])
        f0 = 9.276710625272244e-05
        nz = len(zN2)
        kwargs = {'num_eigen': 2, 'init_vector': None, 'num_Lanczos': nz*10, 'iteration': nz*100, 'tolerance': 0}
        zphi, Rd, v = modes.neutral_modes_from_N2_profile(
                        -zN2, N2, f0, **kwargs)
        
        self.assertTrue(np.isclose(Rd[1], 25787.38588731),
            msg='Rossby radius should be exact to the solution')

        self.assertTrue(np.allclose(np.diff(v[:, 0]), 0.),
            msg='The barotropic vertical mode should have all same value')
        
        v1 = np.array([ 0.2287108,  0.22844163,  0.22738221,  0.22538205,  0.22315635,  0.22100572,
          0.21890111,  0.21681132,  0.21466035,  0.21249153,  0.21051281,  0.20857709,
          0.20653185,  0.20436013,  0.20172795,  0.19864415,  0.19475028,  0.18960952,
          0.18268901,  0.17391394,  0.16270843,  0.14845062,  0.13133273,  0.11305419,
          0.09473064,  0.07691331,  0.05874788,  0.04009015,  0.02387257,  0.01278873,
          0.00637754,  0.00253038, -0.00029387, -0.00275014, -0.00532013, -0.00849032,
         -0.01248929, -0.01713441, -0.02220176, -0.02756655, -0.0331077,  -0.03857259,
         -0.04363333, -0.04770119, -0.05004272, -0.05105026, -0.05168191, -0.05185931])
        self.assertTrue(np.allclose(np.absolute(v[:, 1]), np.absolute(v1)),
            msg='The amplitude of vertical modes should have the exact elements')


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
        
        with self.assertRaises(ValueError):
            _, _, _ = modes.instability_analysis_from_N2_profile(
                -depth, N2, 1., beta, k ,l, np.append(depth, float(nz+1)), ubar, vbar, etax, etay
            )
        
        with self.assertRaises(ValueError):
            _, _, _ = modes.instability_analysis_from_N2_profile(
                depth, N2[1:], 1., beta, k ,l, -np.append(depth, float(nz+1)), ubar, vbar, etax, etay
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
        
        N2 = np.append(N2, [1.,1.]) #N2 has longer length than u
        with self.assertRaises(ValueError):
            _, _, _ = modes.instability_analysis_from_N2_profile(
                depth, N2, 1., beta, k ,l, depth, ubar, vbar, etax, etay
            )
        
        N2 = np.zeros(nz)
        depth = np.arange(nz+1)
        ubar = np.zeros(nz+1)
        vbar = np.zeros(nz+1)
        etax = np.zeros(2)
        etay = np.zeros(2)
        beta = 0.
        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning
            _, _, _ = modes.instability_analysis_from_N2_profile(
                depth[:-1], N2, 0., beta, k ,l, depth, ubar, vbar, etax, etay
            )
            # Verify some things
            self.assertTrue( issubclass(w[-1].category, RuntimeWarning) )
            self.assertEqual("The matrix is ill-conditioned or singular", str(w[-1].message))

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
        
    def test_OCCA_GulfStream(self, atol=1e-8):
        """ Actual profiles in the Gulf Stream region 
             (@ 39.5N, 299.5E)
        """
        ###########
        # prepare parameters for Gulf Stream region in OCCA
        ###########
        z_u = np.array([5.00000000e+00,   1.50000000e+01,   2.50000000e+01,
         3.50000000e+01,   4.50000000e+01,   5.50000000e+01,
         6.50000000e+01,   7.50050049e+01,   8.50250015e+01,
         9.50950012e+01,   1.05309998e+02,   1.15870003e+02,
         1.27150002e+02,   1.39739990e+02,   1.54470001e+02,
         1.72399994e+02,   1.94735001e+02,   2.22710007e+02,
         2.57470001e+02,   2.99929993e+02,   3.50679993e+02,
         4.09929993e+02,   4.77470001e+02,   5.52710022e+02,
         6.34735046e+02,   7.22400024e+02,   8.14470093e+02,
         9.09740112e+02,   1.00715503e+03,   1.10590503e+03,
         1.20553503e+03,   1.30620508e+03,   1.40914990e+03,
         1.51709497e+03,   1.63417480e+03,   1.76513477e+03,
         1.91414990e+03,   2.08403491e+03,   2.27622510e+03,
         2.49125000e+03,   2.72925000e+03,   2.99025000e+03,
         3.27425000e+03,   3.58125000e+03,   3.91125000e+03,
         4.26425000e+03,   4.64025000e+03,   5.03925000e+03,
         5.46125000e+03,   5.90625000e+03])
        z_N2 = -np.array([-10.00006115,   -20.00006114,   -30.00006113,   -40.00006112,
         -50.0000611 ,   -60.00006109,   -70.00256358,   -80.01506451,
         -90.06006328,  -100.20256308,  -110.5900682 ,  -121.51007976,
        -133.44509256,  -147.10512803,  -163.43519352,  -183.5678012 ,
        -208.72298023,  -240.09073926,  -278.70109317,  -325.30655719,
        -380.30712264,  -443.70276088,  -515.09343642,  -593.72659767,
        -678.57216844,  -768.44015906,  -862.11055267,  -958.45325725,
       -1056.53586028, -1155.72595435, -1255.87608991, -1357.68378579,
       -1463.12934261, -1575.64299213, -1699.66489761, -1839.65538686,
       -1999.10931291, -2180.15155508, -2383.76440293, -2610.28273438,
       -2859.78914881, -3132.29607118, -3427.80347989, -3746.3113517 ,
       -4087.81966191, -4452.32838443, -4839.83749199, -5250.34695635,
       -5683.85674858])
        ubar = np.array([0.24113144,  0.20566152,  0.18566666,  0.17763813,  0.17248898,
        0.16799432,  0.16402273,  0.16024273,  0.15672079,  0.1532983 ,
        0.15058691,  0.1480111 ,  0.14513075,  0.14221466,  0.13825534,
        0.13390785,  0.12829159,  0.12135932,  0.11275966,  0.10259634,
        0.09075638,  0.07769765,  0.06402871,  0.05062123,  0.0384005 ,
        0.02791209,  0.01922886,  0.01237531,  0.00755318,  0.00470483,
        0.00328818,  0.00261901,  0.00221545,  0.00182732,  0.0013118 ,
        0.00058111, -0.00035143, -0.0013624 , -0.00230507, -0.00308389,
       -0.00365073, -0.00393951, -0.00386523, -0.00344036, -0.00277483,
       -0.00189117, -0.00060206,  0.00060187,      np.nan,      np.nan])
        vbar = np.array([0.03732778,  0.03306238,  0.03844519,  0.04393799,  0.04580961,
        0.04673327,  0.04731352,  0.04768672,  0.04787091,  0.04791058,
        0.0476704 ,  0.04725287,  0.04666282,  0.04587267,  0.04479391,
        0.04350807,  0.04190299,  0.03983596,  0.03736874,  0.03421749,
        0.03057165,  0.02645037,  0.02181544,  0.01715759,  0.01278938,
        0.00886511,  0.00544434,  0.00260065,  0.00052863, -0.00065227,
       -0.00111283, -0.00120176, -0.00119863, -0.00122544, -0.00130761,
       -0.00143906, -0.00161413, -0.00183596, -0.00210659, -0.00240779,
       -0.002703  , -0.00296008, -0.00314709, -0.00320451, -0.00309561,
       -0.0028392 , -0.00253795,  0.        ,      np.nan,      np.nan])
        N2 = np.array([4.49013733e-05,   1.32619531e-04,   1.67213349e-04,
         1.39980381e-04,   1.08620176e-04,   8.89353316e-05,
         7.59696940e-05,   6.85660114e-05,   6.13053056e-05,
         4.96623307e-05,   4.26153730e-05,   3.83933836e-05,
         3.33244718e-05,   3.14573673e-05,   2.74780120e-05,
         2.50869570e-05,   2.35907064e-05,   2.26047540e-05,
         2.06309696e-05,   1.93499569e-05,   1.85802260e-05,
         1.73887368e-05,   1.49995125e-05,   1.25945779e-05,
         1.06210281e-05,   9.69791903e-06,   9.18845407e-06,
         7.56655382e-06,   5.00383224e-06,   2.83633215e-06,
         1.67170878e-06,   1.19350982e-06,   9.87252978e-07,
         9.54096555e-07,   1.05979705e-06,   1.18957478e-06,
         1.23432375e-06,   1.22190029e-06,   1.20084648e-06,
         1.18287498e-06,   1.14801849e-06,   1.08643614e-06,
         9.36511867e-07,   6.17342678e-07,   3.35507108e-07,
         3.19646379e-07,   2.60733349e-07,              np.nan,
                    np.nan])
        Rd = 25787.38588424
        f0 = 9.27671062527e-05
        beta = 1.76637107718e-11
        Nx = 100; dx = 4e3
        k = np.array([-1.88495559e-04,  -1.72787596e-04,  -1.57079633e-04,  -1.41371669e-04,
          -1.25663706e-04,  -1.09955743e-04,  -9.42477796e-05,  -7.85398163e-05,
          -6.28318531e-05,  -4.71238898e-05,  -3.14159265e-05,  -1.57079633e-05])
        l = np.array([0.])
        etax = np.zeros(2)
        etay = np.zeros(2)

        z, growth_rate, vertical_modes = \
            modes.instability_analysis_from_N2_profile(
                z_N2, N2, f0, beta, k, l,
                z_u, ubar, vbar, etax, etay, num=2
        )
        
        self.assertEqual(len(z), vertical_modes.shape[0],
            msg='modes array must be in the right shape')

        self.assertTrue(np.all( np.diff(
                    growth_rate.reshape((growth_rate.shape[0], 
                                         len(k)*len(l))).imag.max(axis=1) ) <= 0.),
            msg='imaginary part of modes should be descending')
        
        mode_amplitude1 = (np.absolute(vertical_modes[:, 0])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude1),
            msg='mode1 should be normalized to amplitude of 1 at all horizontal wavenumber points')
        
        mode_amplitude2 = (np.absolute(vertical_modes[:, 1])**2).sum(axis=0)
        self.assertTrue(np.allclose(1., mode_amplitude2),
            msg='mode2 should be normalized to amplitude of 1 at all horizontal wavenumber points')
        
        ##########
        # Row corresponding to l=0 of growth mode (Ah=0)
        ##########
        w_sol = np.array([1.04641026e-06,   8.55850224e-07,   6.56131690e-07,   4.40205651e-07,
       2.22218418e-07,   1.07589996e-07,   9.46591756e-08,   4.64407815e-08,
       1.29096295e-07,   6.74206451e-08,   5.74783413e-07,   6.86202530e-08])
        
        self.assertTrue( np.allclose(growth_rate.imag[0, 0], w_sol, atol=atol),
            msg='The OCCA numerical growth rates should be close to the solution with Ah=0' )
        
        z, growth_rate, vertical_modes = \
            modes.instability_analysis_from_N2_profile(
                z_N2, N2, f0, beta, k, l,
                z_u, ubar, vbar, etax, etay, Ah=1e1, num=2
        )
        
        ##########
        # Row corresponding to l=0 of growth mode (Ah=10.)
        ##########
        w_sol = np.array([3.15410778e-06,   2.82584613e-06,   2.47671069e-06,   2.10243465e-06,
       1.69802277e-06,   1.26121608e-06,   8.09682819e-07,   4.13752221e-07,
       1.43811161e-07,   7.84288795e-08,   5.93310629e-07,   6.33544320e-08])
        
        self.assertTrue( np.allclose(growth_rate.imag[0, 0], w_sol, atol=atol),
            msg='The OCCA numerical growth rates should be close to the solution with Ah=10' )