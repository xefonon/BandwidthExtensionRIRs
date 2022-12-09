import numpy as np
from librosa import resample
from glob import glob
from scipy.io import loadmat
import h5py
from sklearn import linear_model
from tqdm import tqdm
import click
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')



def stack_real_imag_H(mat):

    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack

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

def fib_sphere(num_points, radius = 1.):
    ga = (3 - np.math.sqrt(5.)) * np.pi # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points, dtype = np.float32)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1/num_points-1, 1-1/num_points, num_points)

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
    # ax.scatter(x_batch[27, :], y_batch[27, :], z_batch[27, :], s = 3)
    # # ax.scatter(x, y, z , s = 3)
    # plt.show()
    return [x_batch,y_batch,z_batch]

def wavenumber(f, n_PW, T = None):
    if T is None:
        T = 20.
    c = speed_of_sound(T)
    k = 2*np.pi*f/c
    k_grid = fib_sphere(n_PW, k)
    return k_grid

def build_sensing_mat(kx, ky, kz, X, Y, Z, mesh=False):
    if mesh:
        H = np.exp(-1j*(kx.T*X.ravel() + ky.T*Y.ravel() + kz.T*Z.ravel())).T
    else:
        H = np.exp(-1j*(kx.T*X + ky.T*Y + kz.T*Z)).T
    return H

def get_sensing_mat(f, n_pw, X, Y, Z, k_samp=None):
    # Basis functions for coefficients
    if k_samp is None:
        k_samp = wavenumber(f, n_pw)
        # k_samp = random_wavenumber(f, n_pw)

    kx, ky, kz = k_samp
    if np.ndim(kx) < 2:
        kx = np.expand_dims(kx, 0)
        ky = np.expand_dims(ky, 0)
        kz = np.expand_dims(kz, 0)
    k_out = [kx, ky, kz]
    H = build_sensing_mat(kx, ky, kz, X, Y, Z, mesh = True)
    return np.squeeze(H), k_out

def load_and_resample(directory = '../validation_responses', sample_rate = 16000, n_truncation = 16384):

    mat_filepath = glob(directory + '/*.mat')

    for matfile in mat_filepath:
        if matfile.split('.mat')[0].split('iec_')[-1] == 'sphere':
            matfiledict_sphere = loadmat(matfile)
        else:
            matfiledict_reference = loadmat(matfile)

    rirs_sphere = matfiledict_sphere['h'].T
    old_sr = matfiledict_sphere['fs'][0][0]
    rirs_ref = matfiledict_reference['h'].T

    grid_sphere = matfiledict_sphere['r'].T
    grid_ref = matfiledict_reference['r'].T

    rirs_sphere = resample(rirs_sphere, old_sr, sample_rate)
    rirs_ref = resample(rirs_ref, old_sr, sample_rate)
    with h5py.File('IEC_dataset.h5', 'w') as f:
        f['rirs_sphere'] = rirs_sphere[:, :n_truncation]
        f['rirs_ref'] = rirs_ref[:, :n_truncation]
        f['sample_rate'] = sample_rate
        f['grid_sphere'] = grid_sphere
        f['grid_ref'] = grid_ref
        f.close()

def save_h5_from_dict(dictionary, savepath = '../validation_responses/IEC_reconstructions.h5'):
    with h5py.File(savepath, 'w') as f:
        for key in dictionary.keys():
            f[key] = dictionary[key]
        f.close()


def load_dataset(directory = '../validation_responses'):
    dictionary = {}
    filepath = directory + '/IEC_dataset.h5'
    with h5py.File(filepath, 'r') as f:
        keys = f.keys()
        for key in keys:
            try:
                dictionary[key] = f[key][:]
            except:
                dictionary[key] = np.asarray(f[key]).item()
        f.close()
    return dictionary

def normalize(input, norm_ord = np.inf):
    if norm_ord == np.inf:
        norm = np.max(np.abs(input))
    else:
        norm = np.linalg.norm(input, ord = norm_ord, keepdims = True)
    input = input/norm
    return input, norm

def Ridge_regression(H, p, n_plwav):
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
    # alphas=np.logspace(-5, -3, 200)0.02346971
    # alphas = np.geomspace(1e-6, -1, 100)
    alphas=np.geomspace(1e-2, 1e-8, 200)
    # alphas = np.logspace(-5, -3, 200)

    reg = linear_model.RidgeCV(cv=5, alphas=alphas,
                               fit_intercept = True, normalize= True)
    # reg = linear_model.Ridge(alpha=1.2e-6,
    #                          fit_intercept=False, normalize=True)
    # gcv_mode = 'eigen')
    # reg = linear_model.RidgeCV()

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        alpha_titk = 0
        pass
    # Predict
    return q, alpha_titk

def decimate_sphere(points, n_newpoints = 30):
    points_ = points - points.mean(axis = -1)[..., np.newaxis]
    radius = np.max(np.linalg.norm(points_, axis = 0))
    xyz_indices = []
    test_points = np.asarray(fib_sphere(n_newpoints, radius = radius))

    for m in range(test_points.shape[-1]):
        p = test_points[:,m]
        xyz_discrep = np.linalg.norm(p[..., np.newaxis] - points_, axis = 0)
        xyz_indx = xyz_discrep.argmin()
        xyz_indices.append(xyz_indx)
    return points[:,np.unique(xyz_indices)], np.unique(xyz_indices)

def Lasso_regression(H, p, n_plwav):
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
    alphas = np.geomspace(1e-1, 1e-14, 50)
    reg = linear_model.LassoCV(cv=5, alphas=alphas,
                               fit_intercept=True, normalize=True, max_iter = 2000, tol=1e-10, eps = 1e-8)
    # reg = linear_model.LassoLarsCV(cv=10, fit_intercept = True, normalize= True, eps = 1e-18, max_iter = 2000)
    # reg = linear_model.Lasso(alpha=1.2e-6,
    #                          fit_intercept=False, normalize=True)
    # gcv_mode = 'eigen')
    # reg = linear_model.RidgeCV()

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        alpha_titk = 0
        pass
    # Predict
    return q, alpha_titk

def put_together(directory, decimate = False):
    from glob import glob
    lass_files = glob(directory + '/P_lass_*.npy')
    ridge_files = glob(directory + '/P_ridge_*.npy')
    dset = load_dataset()
    freq = np.fft.rfftfreq(n = dset['rirs_sphere'].shape[-1], d = 1/dset['sample_rate'])
    _, norm = normalize(dset['rirs_sphere'])
    rirs_sphere = dset['rirs_sphere']/norm
    rirs_ref = dset['rirs_ref']/norm
    FR_sphere = np.fft.rfft(rirs_sphere)
    FR_ref = np.fft.rfft(rirs_ref)
    n_planewaves = 1200
    # center = dset['grid_sphere'].mean(axis=1)
    grid_sphere = dset['grid_sphere']
    grid_ref = dset['grid_ref']
    hsph = 0.628
    radius = 0.5
    interp_indx = np.squeeze(np.argwhere(np.linalg.norm(grid_ref - np.array([[0., 0., hsph]]).T, axis=0) <= radius - 0.02))
    extrap_index = np.squeeze(np.argwhere(np.linalg.norm(grid_ref - np.array([[0., 0., hsph]]).T, axis=0) > radius - 0.02))

    np.random.seed(1234)
    plus_five = np.random.choice(interp_indx, 5, replace = False)

    FR_fit = np.vstack((FR_sphere, FR_ref[plus_five]))
    grid_fit = np.hstack((grid_sphere, grid_ref[:, plus_five]))
    FR_rec_ridge = np.zeros_like(FR_ref)
    FR_rec_lasso = np.zeros_like(FR_ref)
    for file in lass_files:
        index = int(file.split('P_lass_')[-1].split('.npy')[0])
        P = np.load(file)
        FR_rec_lasso[:, index] = P
    for file in ridge_files:
        index = int(file.split('P_ridge_')[-1].split('.npy')[0])
        P = np.load(file)
        FR_rec_ridge[:, index] = P

    rir_ridge = np.fft.irfft(FR_rec_ridge)/norm
    rir_lass = np.fft.irfft(FR_rec_lasso)/norm
    dset['rir_ridge'] = rir_ridge
    dset['rir_lass'] = rir_lass
    dset['rirs_ref'] = rirs_ref
    dset['rirs_sphere'] = rirs_sphere
    dset['grid_fit'] = grid_fit
    dset['grid_ref'] = grid_ref
    dset['grid_sphere'] = grid_sphere
    dset['n_planewaves'] = n_planewaves
    dset['noise'] = 0.02346971
    dset['T'] = 20
    dset['interp_indx'] = interp_indx
    dset['extrap_index'] = extrap_index
    if decimate:
        save_h5_from_dict(dset, savepath = '../validation_responses/IEC_decimated_array_reconstructions.h5')
    else:
        save_h5_from_dict(dset)
def return_largest_index(directory):
    from glob import glob
    import re
    def extract_number(f):
        s = re.findall("\d+$",f)
        return (int(s[0]) if s else -1,f)

    files = glob(directory + '/*.npy')
    return max(files,key=extract_number)

@click.command()
@click.option('--rec_indx', default=0, type=int,
              help='LSF index for parallel computing, in this context \
              it corresponds to a specific sound field frequency index')
@click.option('--confirm_recon', default=False, type=bool,
              help='Use this variable to check for any missing reconstructions')
@click.option('--decimate', default=False, type=bool,
              help='Use this variable to reconstruct for decimated array (45 microphones)')

def reconstruct(rec_indx, confirm_recon, decimate):
    dset = load_dataset()
    dirname= '../validation_responses/reconstructed_sfs'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if confirm_recon:
        rec_indx = return_largest_index(dirname) + 1
    freq = np.fft.rfftfreq(n = dset['rirs_sphere'].shape[-1], d = 1/dset['sample_rate'])
    _, norm = normalize(dset['rirs_sphere'])

    rirs_sphere = dset['rirs_sphere']
    rirs_ref = dset['rirs_ref']
    grid_sphere = dset['grid_sphere']
    grid_ref = dset['grid_ref']

    if decimate:
        _, decim_indx = decimate_sphere(grid_sphere, n_newpoints = 48)
        grid_sphere = grid_sphere[:, decim_indx]
        rirs_sphere = rirs_sphere[decim_indx]

    FR_sphere = np.fft.rfft(rirs_sphere)
    FR_ref = np.fft.rfft(rirs_ref)
    n_planewaves = 1200
    reconstruct_all = False
    FR_rec_ridge = []
    FR_rec_lasso = []
    recon_alphas_ridge = []
    recon_alphas_lasso = []
    # center = dset['grid_sphere'].mean(axis=1)
    hsph = 0.628
    radius = 0.5
    interp_indx = np.squeeze(np.argwhere(np.linalg.norm(grid_ref - np.array([[0., 0., hsph]]).T, axis=0) <= radius - 0.02))
    extrap_index = np.squeeze(np.argwhere(np.linalg.norm(grid_ref - np.array([[0., 0., hsph]]).T, axis=0) > radius - 0.02))

    np.random.seed(1234)
    plus_five = np.random.choice(interp_indx, 5, replace = False)

    FR_fit = np.vstack((FR_sphere, FR_ref[plus_five]))
    grid_fit = np.hstack((grid_sphere, grid_ref[:, plus_five]))

    if reconstruct_all:
        pbar = tqdm(freq)
        for ii, f in enumerate(pbar):
            # f = 250
            # ii = np.argmin(freq< f)
            Pvec = FR_fit[:, ii]
            H, k = get_sensing_mat(f,n_planewaves, grid_fit[0], grid_fit[1], grid_fit[2])
            # split into real + imaginary
            Hridge = stack_real_imag_H(H)
            # Squeeze pressure from tensor + seperate pressure into [real, imaginary]
            Pm_ls = np.concatenate((Pvec.real, Pvec.imag))
            # Run regression algorithm
            qridge, alphas = Ridge_regression(Hridge, Pm_ls, n_planewaves)
            qlass, alphas_lass = Lasso_regression(Hridge, Pm_ls, n_planewaves)
            # Get projection sensing matrix (project onto reference plane)
            Hextrap, _ = get_sensing_mat(f, n_planewaves,grid_ref[0], grid_ref[1], grid_ref[2], k_samp=k)

            # predict pressure p = Hx
            P_ridge = Hextrap @ qridge
            P_lass = Hextrap @ qlass
            FR_rec_ridge.append(P_ridge)
            FR_rec_lasso.append(P_lass)
            recon_alphas_ridge.append(alphas)
            recon_alphas_lasso.append(alphas_lass)
            pbar.set_postfix({'frequency' : f})
        rir_ridge = np.fft.irfft(np.array(FR_rec_ridge).T)
        rir_lass = np.fft.irfft(np.array(FR_rec_lasso).T)
        dset['rir_ridge'] = rir_ridge
        dset['rir_lass'] = rir_lass
        dset['alphas_ridge'] = np.array(recon_alphas_ridge)
        dset['alphas_lasso'] = np.array(recon_alphas_lasso)
        dset['rirs_ref'] = rirs_ref
        dset['rirs_sphere'] = rirs_sphere
        dset['grid_fit'] = grid_fit
        dset['grid_ref'] = grid_ref
        dset['grid_sphere'] = grid_sphere
        dset['n_planewaves'] = n_planewaves
        dset['noise'] = 0.02346971
        dset['T'] = 18
        dset['interp_indx'] = interp_indx
        dset['extrap_index'] = extrap_index
        save_h5_from_dict(dset)

    else:
        dirname= '../validation_responses/reconstructed_sfs'
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        ii = rec_indx
        f = freq[rec_indx]
        Pvec = FR_fit[:, ii]
        H, k = get_sensing_mat(f,n_planewaves, grid_fit[0], grid_fit[1], grid_fit[2])
        # split into real + imaginary
        Hridge = stack_real_imag_H(H)
        # Squeeze pressure from tensor + seperate pressure into [real, imaginary]
        Pm_ls = np.concatenate((Pvec.real, Pvec.imag))
        # Run regression algorithm
        qridge, alphas = Ridge_regression(Hridge, Pm_ls, n_planewaves)
        qlass, alphas_lass = Lasso_regression(Hridge, Pm_ls, n_planewaves)
        # Get projection sensing matrix (project onto reference plane)
        Hextrap, _ = get_sensing_mat(f, n_planewaves,grid_ref[0], grid_ref[1], grid_ref[2], k_samp=k)

        # predict pressure p = Hx
        P_ridge = Hextrap @ qridge
        P_lass = Hextrap @ qlass
        # FR_rec_ridge[:, rec_indx] = P_ridge
        # FR_rec_lasso[:, rec_indx] = P_lass
        recon_alphas_ridge.append(alphas)
        recon_alphas_lasso.append(alphas_lass)
        # dset['FR_rec_ridge'] = FR_rec_ridge
        # dset['FR_rec_lasso'] = FR_rec_lasso
        print(f"Reconstructed for index: {rec_indx}")
        if rec_indx == 0:
            dset['rirs_ref'] = rirs_ref
            dset['rirs_sphere'] = rirs_sphere
            dset['grid_fit'] = grid_fit
            dset['grid_ref'] = grid_ref
            dset['grid_sphere'] = grid_sphere
            dset['n_planewaves'] = n_planewaves
            dset['noise'] = 0.02346971
            dset['T'] = 18
            dset['interp_indx'] = interp_indx
            dset['extrap_index'] = extrap_index
            save_h5_from_dict(dset)

        np.save(dirname + f'/P_ridge_{rec_indx}', P_ridge)
        np.save(dirname + f'/P_lass_{rec_indx}', P_lass)

if __name__ == '__main__':
    dirname= '../validation_responses/reconstructed_sfs'
    put_together(dirname, decimate = True)