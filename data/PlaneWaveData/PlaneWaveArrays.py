import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import h5py
from scipy.stats import halfnorm
import click
import os
from sklearn import linear_model
import time
import warnings
from scipy.linalg import LinAlgWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')


def standardize_data_vector(data):
    return (data - data.mean()) / data.std()


def example_of_reconstruction(f=250, cv=True):
    pm, pref, grid, grid_ref, pw_obj = synth_plane_wave_sf(100, 0.6, 20, f, return_grids=True)

    H, k = pw_obj.get_sensing_mat(f, 3000,
                                  grid[0],
                                  grid[1],
                                  grid[2])

    Href, _ = pw_obj.get_sensing_mat(f, 3000,
                                     grid_ref[0],
                                     grid_ref[1],
                                     grid_ref[2],
                                     k_samp=k)

    startlars = time.time()
    coeffs_larsLasso, alpha_larsLasso = LASSOLARS_regression(np.squeeze(H), pm, cv=cv)
    endlars = time.time()
    startridge = time.time()
    coeffs_ridge, alpha_ridge = Ridge_regression(np.squeeze(H), pm, cv=cv)
    endridge = time.time()
    startlass = time.time()
    coeffs_lasso, alpha_lasso = LASSO_regression(np.squeeze(H), pm, cv=cv)
    endlass = time.time()

    rmse = lambda x, y: np.sqrt((abs(y - x) ** 2).mean())

    plars = np.squeeze(Href) @ coeffs_larsLasso
    plass = np.squeeze(Href) @ coeffs_lasso
    pridge = np.squeeze(Href) @ coeffs_ridge
    print("alpha Lars Lasso: {}, time: {:.4f}, error: {:.5f}".format(alpha_larsLasso, -(startlars - endlars),
                                                                     rmse(plars, pref)))
    print("alpha Ridge: {}, time: {:.4f}, error: {:.5f}".format(alpha_ridge, -(startridge - endridge),
                                                                rmse(pridge, pref)))
    print("alpha Lasso: {}, time: {:.4f}, error: {:.5f}".format(alpha_lasso, -(startlass - endlass), rmse(plass, pref)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    pw_obj.plot_array_pressure(pref, grid_ref, ax=ax)
    ax.set_title('truth')
    ax = fig.add_subplot(1, 4, 2, projection='3d')
    pw_obj.plot_array_pressure(plars, grid_ref, ax=ax)
    ax.set_title('Lars')
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    pw_obj.plot_array_pressure(plass, grid_ref, ax=ax)
    ax.set_title('Lasso')
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    pw_obj.plot_array_pressure(pridge, grid_ref, ax=ax)
    ax.set_title('Ridge')

    fig.show()


def Ridge_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1e-2, 1e-7, 50),
                                   fit_intercept=True, normalize=True)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept=True, normalize=True)

    # gcv_mode = 'eigen')
    # reg = linear_model.RidgeCV()
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk


def LASSO_regression(H, p, n_plwav=None, cv=True):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    if cv:
        reg_las = linear_model.LassoCV(cv=5, alphas=np.geomspace(1e-7, 1e-2, 50),
                                       fit_intercept=True, normalize=True)
    else:
        alpha_lass = 2.62e-6
        reg_las = linear_model.Lasso(alpha=alpha_lass,
                                     fit_intercept=True, normalize=True)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        pass
    return q_las, alpha_lass


def LASSOLARS_regression(H, p, n_plwav=None, cv=True):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    if cv:
        reg_las = linear_model.LassoLarsCV(cv=5, fit_intercept=True, normalize=True)
    else:
        alpha_lass = 2.62e-6
        reg_las = linear_model.LassoLars(alpha=alpha_lass, fit_intercept=True, normalize=True)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    try:
        alpha_lass = reg_las.alpha_
    except:
        pass

    return q_las, alpha_lass


def OrthoMatchPursuit_regression(H, p, n_plwav=None):
    """
    Compressive Sensing - Soundfield Reconstruction

    Parameters
    ----------
    H : Transfer Matrix.
    p : Measured Pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q_las : Plane wave coefficients.
    alpha_lass : Regularizor.

    """
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if H.dtype == complex:
        H = stack_real_imag_H(H)
    if p.dtype == complex:
        p = np.concatenate((p.real, p.imag))

    reg_las = linear_model.OrthogonalMatchingPursuitCV(cv=5, max_iter=1e4)
    # reg_las = linear_model.LassoLarsCV( )
    # alphas = np.logspace(-14, 2, 17))#cv=5, max_iter = 1e5, tol=1e-3)

    reg_las.fit(H, p)
    q_las = reg_las.coef_[:n_plwav] + 1j * reg_las.coef_[n_plwav:]
    return q_las


def modal_overlap(Trev, V, c, freq):
    return 12. * np.math.log(10.) * V * freq ** 2 / (Trev * c ** 3)


def number_plane_waves(Trev, V, c, freq):
    """
    Number of plane waves according to oblique modes in a room
    e.g. F. Jacobsen "Fundamentals of general linear acoustics" pp. 137-140
    Parameters
    ----------
    Trev
    V
    c
    freq

    Returns
    -------

    """
    M = modal_overlap(Trev, V, c, freq)
    return 8 * M


@jit(nopython=True)
def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack


@jit(nopython=True)
def _build_sensing_mat(kx, ky, kz, X, Y, Z):
    # H = np.exp(-1j * (np.einsum('ij,k -> ijk', kx, X) + \
    #                   np.einsum('ij,k -> ijk', ky, Y) + \
    #                   np.einsum('ij,k -> ijk', kz, Z)))
    # return np.transpose(H, axes=[0, 2, 1])
    H = np.exp(-1j * (np.outer(kx, X) + \
                      np.outer(ky, Y) + \
                      np.outer(kz, Z)))
    H = np.expand_dims(H, axis=0)
    return H


# @jit(nopython=True)
def adjustSNR(sig, snrdB=40, td=True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from signal
    ndim = sig.ndim
    if ndim > 2:
        dims = (-2, -1)
    else:
        dims = -1
    mean = np.mean(sig, axis=dims)
    # remove DC
    if ndim > 2:
        sig_zero_mean = sig - mean[..., np.newaxis, np.newaxis]
    else:
        sig_zero_mean = sig - mean[..., np.newaxis]

    var = np.var(sig_zero_mean, axis=dims)
    if ndim > 2:
        psig = var[..., np.newaxis, np.newaxis]
    else:
        psig = var[..., np.newaxis]

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    pnoise = psig / snr_lin

    if td:
        # Create noise vector
        noise = np.sqrt(pnoise) * np.random.randn(sig.shape)
    else:
        # complex valued white noise
        real_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        imag_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        noise = real_noise + 1j * imag_noise
        noise_mag = np.sqrt(pnoise) * np.abs(noise)
        noise = noise_mag * np.exp(1j * np.angle(noise))

    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise


# @jit(nopython=True)
def fib_sphere(num_points, radius=1.):
    ga = (3 - np.sqrt(5.)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * np.sin(theta)
    x = alpha * np.cos(theta)

    x_batch = np.dot(radius, x)
    y_batch = np.dot(radius, y)
    z_batch = np.dot(radius, z)

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 3)
    # plt.show()
    return np.asarray([x_batch, y_batch, z_batch])


@jit(nopython=True)
def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the adiabatic speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * np.sqrt(273.15 + T)
    return c


@jit(nopython=True)
def _cart2sph(x, y, z):
    r"""Cartesian to spherical coordinate transform.
    .. math::
        \alpha = \arctan \left( \frac{y}{x} \right) \\
        \beta = \arccos \left( \frac{z}{r} \right) \\
        r = \sqrt{x^2 + y^2 + z^2}
    with :math:`\alpha \in [-pi, pi], \beta \in [0, \pi], r \geq 0`
    Parameters
    ----------
    x : float or array_like
        x-component of Cartesian coordinates
    y : float or array_like
        y-component of Cartesian coordinates
    z : float or array_like
        z-component of Cartesian coordinates
    Returns
    -------
    theta : float or `numpy.ndarray`
            Azimuth angle in radians
    phi : float or `numpy.ndarray`
            Colatitude angle in radians (with 0 denoting North pole)
    r : float or `numpy.ndarray`
            Radius
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi, r


@jit(nopython=True)
def _sph2cart(alpha, beta, r):
    """Spherical to cartesian coordinate transform.
    .. math::
        x = r \cos \alpha \sin \beta \\
        y = r \sin \alpha \sin \beta \\
        z = r \cos \beta
    with :math:`\alpha \in [0, 2\pi), \beta \in [0, \pi], r \geq 0`
    Parameters
    ----------
    alpha : float or array_like
            Azimuth angle in radiants
    beta : float or array_like
            Colatitude angle in radiants (with 0 denoting North pole)
    r : float or array_like
            Radius
    Returns
    -------
    x : float or `numpy.ndarray`
        x-component of Cartesian coordinates
    y : float or `numpy.ndarray`
        y-component of Cartesian coordinates
    z : float or `numpy.ndarray`
        z-component of Cartesian coordinates
    """
    x = r * np.cos(alpha) * np.sin(beta)
    y = r * np.sin(alpha) * np.sin(beta)
    z = r * np.cos(beta)
    return x, y, z


@jit(nopython=True)
def _disk_grid_fibonacci(n, r, c, z=None):
    """
    Get circular disk grid points
    Parameters
    ----------
    n : integer N, the number of points desired.
    r : float R, the radius of the disk.
    c : tuple of floats C(2), the coordinates of the center of the disk.
    z : float (optional), height of disk
    Returns
    -------
    cg :  real CG(2,N) or CG(3,N) if z != None, the grid points.
    """
    r0 = r / np.sqrt(float(n) - 0.5)
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    gr = np.zeros(n)
    gt = np.zeros(n)
    for i in range(0, n):
        gr[i] = r0 * np.sqrt(i + 0.5)
        gt[i] = 2.0 * np.pi * float(i + 1) / phi

    if z is None:
        cg = np.zeros((3, n))
    else:
        cg = np.zeros((2, n))

    for i in range(0, n):
        cg[0, i] = c[0] + gr[i] * np.cos(gt[i])
        cg[1, i] = c[1] + gr[i] * np.sin(gt[i])
        if z != None:
            cg[2, i] = z
    return cg


@jit(nopython=True)
def propagation_matmul(H, x):
    # return np.einsum('ijk, ik -> ij', H, x)
    return H @ x


class planewaves:
    def __init__(self, T=20, n_mics=150, radius=1.):
        self.T = T
        self.c = speed_of_sound(T)  # speed of sound for ambient conditions
        self.n_mics = n_mics
        self.radius = radius

    def pure_tone(self, f=1000):
        self.f = f  # frequency
        self.omega = 2 * np.pi * self.f  # angular frequency
        self.k = self.omega / self.c  # wavenumber

    def transfer_fun(self, Nfft=8192, sampling_rate=16000):
        self.freq = np.fft.rfftfreq(int(2 * Nfft), 1 / sampling_rate)
        self.omega = 2 * np.pi * self.freq  # angular frequency
        self.k = self.omega / self.c  # wavenumber

    def spacing(self, start, stop, step=1, *, endpoint=False, dtype=None,
                **kwargs):
        """Like :func:`numpy.arange`, but compensating numeric errors.
        Unlike :func:`numpy.arange`, but similar to :func:`numpy.linspace`,
        providing ``endpoint=True`` includes both endpoints.
        Parameters
        ----------
        start, stop, step, dtype
            See :func:`numpy.arange`.
        endpoint
            See :func:`numpy.linspace`.
            .. note:: With ``endpoint=True``, the difference between *start*
            and *end* value must be an integer multiple of the
            corresponding *spacing* value!
        **kwargs
            All further arguments are forwarded to :func:`numpy.isclose`.
        Returns
        -------
        `numpy.ndarray`
            Array of evenly spaced values.  See :func:`numpy.arange`.
        """
        remainder = (stop - start) % step
        if np.any(np.isclose(remainder, (0.0, step), **kwargs)):
            if endpoint:
                stop += step * 0.5
            else:
                stop -= step * 0.5
        elif endpoint:
            raise ValueError("Invalid stop value for endpoint=True")
        return np.arange(start, stop, step, dtype)

    def direction_vector(self, alpha, beta=np.pi / 2):
        """Compute normal vector from azimuth, colatitude.
            alpha: scalar/float - azimuth angle
            beta: scalar/float - colatitude angle. Default is np.pi/2.
        """
        return self.sph2cart(alpha, beta, 1)

    def sph2cart(self, alpha, beta, r):
        return _sph2cart(alpha, beta, r)

    def cart2sph(self, x, y, z):
        return _cart2sph(x, y, z)

    def disk_grid_fibonacci(self, n, r, c, z=None):
        return _disk_grid_fibonacci(n, r, c, z)

    def mesh_XYZ(self, x_, y_, z_, *, delta=.1, endpoint=True, **kwargs):
        """Create a grid with given range and spacing.
        Parameters
        ----------
        x, y, z : float or pair of float
            Inclusive range of the respective coordinate or a single value
            if only a slice along this dimension is needed.
        delta : float or triple of float
            Grid spacing.  If a single value is specified, it is used for
            all dimensions, if multiple values are given, one value is used
            per dimension.  If a dimension (*x*, *y* or *z*) has only a
            single value, the corresponding spacing is ignored.
        endpoint : bool, optional
            If ``True`` (the default), the endpoint of each range is
            included in the grid.  Use ``False`` to get a result similar to
            :func:`numpy.arange`.  See `strict_arange()`.
        **kwargs
            All further arguments are forwarded to `strict_arange()`.
        Returns
        -------
        `XyzComponents`
            A grid that can be used for sound field calculations.
        See Also
        --------
        strict_arange, numpy.meshgrid
        """
        if np.isscalar(delta):
            delta = [delta] * 3
        ranges = []
        scalars = []
        for i, coord in enumerate([x_, y_, z_]):
            if np.isscalar(coord):
                scalars.append((i, coord))
            else:
                start, stop = coord
                ranges.append(self.spacing(start, stop, delta[i],
                                           endpoint=endpoint, **kwargs))
        Y, X, Z = np.meshgrid(*ranges, sparse=True, copy=False)
        # for i, s in scalars:
        #     grid.insert(i, s)
        return X, Y, Z

    def plane(self, x0, k0, grid=None):
        """
        x0 : (3,) array_like
            Position of plane wave.
        k0 : (3,) array_like
            Normal vector (direction) of plane wave.
        grid : triple of array_like
            The grid that is used for the sound field calculations.
        """

        if grid is None:
            AssertionError("grid must be provided with meshXYZ method")
        k0 = k0 / np.linalg.norm(k0)
        self.P = np.exp(-1j * self.k * np.inner(grid - x0, k0))
        return self.P

    def plot_sf(self, X, Y, P=None, f=None, ax=None, name=None, save=False, add_meas=None,
                clim=(-1, 1), tex=False):
        """
        Plot spatial soundfield normalised amplitude
        --------------------------------------------
        Args:
            P : Pressure in meshgrid [X,Y]
            X : X mesh matrix
            Y : Y mesh matrix
        Returns:
            ax : pyplot axes (optionally)
        """
        cmap = 'coolwarm'
        if P is None:
            P = self.P
        if f is None:
            f = self.f
        P = P / np.max(abs(P))
        x = X.flatten()
        y = Y.flatten()
        if tex:
            plt.rc('text', usetex=True)
        # x, y = X, Y

        dx = 0.5 * x.ptp() / P.size
        dy = 0.5 * y.ptp() / P.size
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        im = ax.imshow(np.real(P), cmap=cmap, origin='lower',
                       extent=[x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy])
        ax.set_ylabel('y [m]')
        ax.set_xlabel('x [m]')
        lm1, lm2 = clim
        im.set_clim(lm1, lm2)
        cbar = plt.colorbar(im)
        # cbar.ax.get_yaxis().labelpad = 15
        # cbar.ax.set_ylabel('Normalised SPL [dB]', rotation=270)
        if add_meas is not None:
            x_meas = X.ravel()[add_meas]
            y_meas = Y.ravel()[add_meas]
            ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

        if name is not None:
            ax.set_title(name + ' - f : {} Hz'.format(f))
        if save:
            plt.savefig(name + '_plot.png', dpi=150)
        return ax

    def plot_3dsf(self, p, mesh=None, ax=None):
        import matplotlib as mpl
        if mesh == None:
            x, y, z = self.mesh_XYZ((-1.28, 1.28), (-1.28, 1.28), (-1.28, 1.28), delta=.04, endpoint=False)
            res = p.shape[0]
            res = np.complex(0, res)
            Xx, Yy, Zz = np.mgrid[x.min():x.max():res, y.min():y.max():res, z.min():z.max():res]
        else:
            Xx, Yy, Zz = mesh
        cmp = plt.get_cmap("cividis")
        if ax == None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

        norm = mpl.colors.Normalize(vmin=p.real.min(), vmax=p.real.max())

        ax.scatter(Xx, Yy, Zz, c=p.real.ravel(), cmap=cmp, alpha=.04, s=2)
        # equal aspect ratio
        ax.set_box_aspect((Xx.max(), Yy.max(), Zz.max()))

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=ax, label='Sound Pressure [Pa]', pad=.06)
        # fig.tight_layout()
        return fig, ax

    def plot_array_pressure(self, p_array, array_grid, ax=None, plane=False, norm=None, cmap=None):
        if ax is None:
            ax = plt.axes(projection='3d')

        if cmap is None:
            cmp = plt.get_cmap("cividis")
        else:
            cmp = plt.get_cmap(cmap)

        if norm is None:
            vmin = p_array.real.min()
            vmax = p_array.real.max()
        else:
            vmin, vmax = norm
        sc = ax.scatter(array_grid[0], array_grid[1], array_grid[2], c=p_array.real,
                        cmap=cmp, alpha=1., s=200, vmin=vmin, vmax=vmax)
        # ax.view_init(90, 90)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        if plane:
            ax.set_box_aspect((1, 1, 1))
        else:
            ax.set_box_aspect((array_grid[0].max(), array_grid[1].max(), array_grid[2].max()))
        return ax, sc

    def get_spherical_array(self):
        array = fib_sphere(self.n_mics, self.radius)
        # validation grid
        grid_ref = self.disk_grid_fibonacci(800, self.radius + .4 * self.radius, (0., 0.))
        x_ref, y_ref, z_ref = grid_ref
        # number of interior points for zero-cross of bessel functions
        npoints = 5
        mask = np.argwhere(x_ref.ravel() ** 2 + y_ref.ravel() ** 2 <= self.radius ** 2)
        interp_ind = np.random.choice(mask.shape[0], size=npoints, replace=False)
        interp_ind = np.squeeze(mask[interp_ind])
        # add 5 points inside the array to avoid singularities
        gridsphere = np.hstack((array, np.asarray([x_ref[interp_ind],
                                                   y_ref[interp_ind],
                                                   z_ref[interp_ind]])
                                ))
        return gridsphere

    def wavenumber(self, f=1000, n_PW=2000, c=343):
        k = 2 * np.pi * f / c
        k_grid = fib_sphere(int(n_PW), k)
        return k_grid

    def build_sensing_mat(self, kx, ky, kz, X, Y, Z, mesh=False):
        H = _build_sensing_mat(kx, ky, kz, X, Y, Z)
        return np.transpose(H, axes=[0, 2, 1])

    def get_sensing_mat(self, f, n_pw, X, Y, Z, k_samp=None):
        # Basis functions for coefficients
        if k_samp is None:
            k_samp = self.wavenumber(f, n_pw, speed_of_sound(self.T))
        kx, ky, kz = k_samp
        if kx.ndim < 2:
            kx = kx[np.newaxis, ...]
            ky = ky[np.newaxis, ...]
            kz = kz[np.newaxis, ...]
        k_out = [kx, ky, kz]
        H = self.build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=False)
        return H, k_out

    def simulate_measurements(self, grid, grid_ref, f=1000, n_plane_waves=2000,
                              n_active_waves=1, snr=None, mag_dist='gamma', return_pw_dict=False,
                              mag_phase_mean=None):

        H, k = self.get_sensing_mat(f, n_plane_waves,
                                    grid[0],
                                    grid[1],
                                    grid[2])
        Href, _ = self.get_sensing_mat(f,
                                       n_plane_waves,
                                       grid_ref[0],
                                       grid_ref[1],
                                       grid_ref[2],
                                       k_samp=k)
        if mag_phase_mean is None:
            pw_phase = np.random.uniform(0, 2 * np.pi, size=(1, n_plane_waves))
            if mag_dist == 'gamma':
                pw_mag = np.random.gamma(0.5, 1, size=(1, n_plane_waves))
            else:
                pw_mag = np.random.uniform(0, 1, size=(1, n_plane_waves))

            if n_active_waves != -1:
                indices = np.arange(0, n_plane_waves, dtype=int)
                indx = np.random.choice(indices, size=n_plane_waves - n_active_waves, replace=False)
                pw_mag[:, indx] = 0.
            mag_phase_mean = [pw_mag, pw_phase]
        else:
            mag_mu, phase_mu = mag_phase_mean
            # eq. 8.65 pp 149 Fundamentals of genaral linear acoustics
            std = np.sqrt(2 / (modal_overlap(self.Trev, self.V, self.c, f) * np.pi))

            if f > self.fsch:  # perfectly diffuse
                pw_phase = np.random.uniform(0, 2 * np.pi, size=(1, n_plane_waves))
            else:  # below schroeder frequency
                pw_phase = np.random.normal(phase_mu, std)
            pw_mag = np.random.normal(mag_mu, std)
            pw_mag[pw_mag < 0.] = 0.
            if pw_mag.shape[-1] < n_plane_waves:
                stack_len = n_plane_waves - pw_mag.shape[-1]
                while mag_mu.shape[-1] < stack_len:
                    mag_mu = np.hstack((mag_mu, mag_mu))
                pw_mag = np.hstack((pw_mag, np.random.normal(mag_mu[:, :stack_len], std)))
            if pw_phase.shape[-1] < n_plane_waves:
                stack_len = n_plane_waves - pw_phase.shape[-1]
                while phase_mu.shape[-1] < stack_len:
                    phase_mu = np.hstack((phase_mu, phase_mu))
                pw_phase = np.hstack((pw_phase, np.random.normal(phase_mu[:, :stack_len], std)))

        pwcoeff = pw_mag * np.exp(1j * pw_phase)
        # if pw_mag
        # pm = np.einsum('ijk, ik -> ij', H, pwcoeff)
        pm = propagation_matmul(np.squeeze(H, axis=0), np.squeeze(pwcoeff, axis=0))
        # pref = np.einsum('ijk, ik -> ij', Href, pwcoeff)
        pref = propagation_matmul(np.squeeze(Href, axis=0), np.squeeze(pwcoeff, axis=0))

        if snr is not None:
            pm = adjustSNR(pm, snrdB=snr, td=False)

        if return_pw_dict:
            return pm / n_plane_waves, pref / n_plane_waves, H, Href, k
        else:
            return pm / n_plane_waves, pref / n_plane_waves, mag_phase_mean

    def schroeder_freq(self):
        return np.sqrt(self.c ** 3 * self.Trev / (4 * np.log(10) * self.V))

    def frequency_response(self, include_reconstruction=False, reconstruction_method='ridge'):

        max_plane_waves = 4000
        Trev = np.random.uniform(0.01, 0.6)
        self.Trev = Trev
        V = np.random.uniform(75, 115)
        self.V = V
        self.fsch = self.schroeder_freq()
        grid = self.get_spherical_array()
        grid_ref = self.disk_grid_fibonacci(n=800, r=self.radius + .4 * self.radius, c=(0., 0.))
        snr = 32 + 5 * np.random.randn()
        measurement = {}
        measurement['pm'] = []
        measurement['pref'] = []
        measurement['f'] = []
        if include_reconstruction:
            measurement['prec'] = []
        pbar = tqdm(self.freq)
        for i, ff in enumerate(pbar):
            n_pw = number_plane_waves(Trev, V, self.c, ff)
            if n_pw <= 8:
                n_pw = 8
            n_pw = np.minimum(np.ceil(n_pw),
                              max_plane_waves)  # max of 4k plane waves for computational purposes
            if i == 0:
                mag_phase_mean = None
            pm, pref, mag_phase_mean = self.simulate_measurements(grid, grid_ref, f=ff, n_plane_waves=int(n_pw),
                                                                  n_active_waves=-1, snr=snr, mag_dist='uniform',
                                                                  mag_phase_mean=None)
            if ff == 0.:
                pm, pref = pm.real + 1j * 0., pref.real + 1j * 0.

            if include_reconstruction:
                H, k = self.get_sensing_mat(ff,
                                            max_plane_waves,
                                            grid[0],
                                            grid[1],
                                            grid[2])

                Href, _ = self.get_sensing_mat(ff,
                                               max_plane_waves,
                                               grid_ref[0],
                                               grid_ref[1],
                                               grid_ref[2],
                                               k_samp=k)

                if reconstruction_method == 'ridge':
                    coeffs, alpha_ = Ridge_regression(np.squeeze(H), pm, cv=False)
                    prec = Href @ coeffs
                elif reconstruction_method == 'lasso':
                    coeffs, alpha_ = LASSO_regression(np.squeeze(H), pm, cv=False)
                    prec = Href @ coeffs
                elif reconstruction_method == 'larslasso':
                    coeffs, alpha_ = LASSOLARS_regression(np.squeeze(H), pm, cv=False)
                    prec = Href @ coeffs
                else:
                    raise ValueError(f'Cannot reconstruct with method {reconstruction_method}, please use "ridge",'
                                     f' "lasso" or "larslasso".')

                measurement['prec'].append(np.squeeze(prec))
            measurement['pm'].append(pm)
            measurement['pref'].append(pref)
            measurement['f'].append(ff)
            pbar.set_postfix({"Frequency": ff})
        measurement['grid'] = grid
        measurement['grid_ref'] = grid_ref
        measurement['snr'] = snr
        measurement['Trev'] = Trev
        measurement['V'] = V

        return measurement


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        for key, item in dic.items():
            if isinstance(item, list):
                h5file[key] = np.asarray(item)
            else:
                h5file[key] = item
        h5file.close()


def save_paired_responses(dic, filepath, index, sample_len=16384):
    with open('sound_field_metadata.txt', 'a') as f:
        f.write(80 * '=' + '\n')
        f.write(f'Sound Field {index}\n')
        f.write(80 * '=' + '\n')
        for key, value in dic.items():
            if isinstance(value, (list, np.ndarray)):
                if key == 'pm':
                    f.write('n_mics:%s\n' % (np.array(value).shape[-1]))
                else:
                    pass
            else:
                f.write('%s:%s\n' % (key, value))
        f.close()
    for key, item in dic.items():
        if key == 'prec':
            prec = np.fft.irfft(np.asarray(item).T, n=sample_len)
        if key == 'pref':
            pref = np.fft.irfft(np.asarray(item).T, n=sample_len)
    for ii in range(len(pref)):
        np.savez_compressed(filepath + f'/responses_sf_{index}_{ii}', pref=pref[ii], prec=prec[ii])


def load_dict_from_hdf5(filename):
    """
    ....
    """
    new_dict = {}
    with h5py.File(filename, "r") as file:
        for f in file.keys():
            new_dict[f] = np.asarray(file[f])
        file.close()
        return new_dict


@click.command()
@click.option('--init_n_mics', default=102, type=int,
              help='Initial number of microphones in spherical \
                    array. If array_base = False, this is the \
                    number of microphones in array, otherwise \
                    the bottom of the array is removed')
@click.option('--radius', default=.5, type=float,
              help='Spherical array radius')
@click.option('--data_dir', default='./SoundFieldData', type=str,
              help='Directory to save synthesised sound fields')
@click.option('--lsf_index', default=23, type=int,
              help='LSF index for parallel computing, in this context \
              it corresponds to a specific sound field')
def synth_sound_fields(init_n_mics, radius, data_dir, lsf_index, save_as_soundfield=False):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # seed the run
    np.random.seed(lsf_index)
    # init plane waves object
    rad = halfnorm(loc=radius / 2, scale=.5).rvs()  # halfnormal centred around 'radius' variable
    temperature = 20 + np.random.randn()
    # λ (init_n_mics here) is the expected rate of occurrences for the poisson distribution
    n_mics = np.random.poisson(init_n_mics)
    pw_obj = planewaves(T=temperature, n_mics=n_mics, radius=rad)
    # tell plane wave object that you need a frequency response of random waves
    # and initialise parameters
    pw_obj.transfer_fun(sampling_rate=16000, Nfft=8192)
    # this is where the magic happens, wait for object to synthesise responses
    meas = pw_obj.frequency_response(include_reconstruction=True, reconstruction_method='ridge')
    if save_as_soundfield:
        save_dict_to_hdf5(meas, data_dir + f'/SoundField_{lsf_index}.h5')
    else:
        save_paired_responses(meas, data_dir, lsf_index, 16384)


def synth_plane_wave_sf(init_n_mics, radius, lsf_index, frequency=200, return_grids=False):
    # seed the run
    np.random.seed(lsf_index)
    # init plane waves object
    rad = radius
    temperature = 20 + np.random.randn()
    # λ (init_n_mics here) is the expected rate of occurrences for the poisson distribution
    n_mics = init_n_mics
    pw_obj = planewaves(T=temperature, n_mics=n_mics, radius=rad)
    # tell plane wave object that you need a single frequency pressure field composed of random waves
    # and initialise parameters
    pw_obj.pure_tone(f=frequency)
    # this is where the magic happens, wait for object to synthesise responses
    Trev = np.random.uniform(0.08, 1.5)
    V = np.random.uniform(25, 50)
    grid = pw_obj.get_spherical_array()
    grid_ref = pw_obj.disk_grid_fibonacci(n=800, r=pw_obj.radius + .4 * pw_obj.radius, c=(0., 0.))
    snr = 32 + 5 * np.random.randn()
    n_pw = number_plane_waves(Trev, V, pw_obj.c, frequency)
    n_pw = np.minimum(np.ceil(n_pw + 1e-12), 4000)  # max of 4k plane waves for computational purposes
    pm, pref, H, Href, k = pw_obj.simulate_measurements(grid, grid_ref, f=frequency, n_plane_waves=int(n_pw),
                                                        n_active_waves=-1, snr=snr, mag_dist='uniform',
                                                        return_pw_dict=True)
    if return_grids:
        return pm, pref, grid, grid_ref, pw_obj
    else:
        return pm, pref


if __name__ == '__main__':
    synth_sound_fields()
