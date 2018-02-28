import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from scipy.interpolate import griddata, interp2d
from sklearn.preprocessing import scale


def reps_to_array(reps):
    """
    reps: dictionary containing the reps for each variation level
    """
    max_reps = np.max([reps[i].shape[0] for i in reps.keys()], axis=0)
    hvm_neural = np.zeros((max_reps, 5760, reps['V0'].shape[2]))
    hvm_neural[...] = np.NaN

    c = 0
    for key in reps:
        shape = reps[key].shape
        hvm_neural[:shape[0], c:c+shape[1], :] = reps[key]
        c += shape[1]
    return hvm_neural


def fix_nan_reps(reps):
    """Some of the entries in neural reps might be nan.
    Substitute those values by the average response of
    corresponding neurons to all images over all valid reps.
    reps = [n_reps, n_samples, n_neurons]
    """
    if np.any(np.isnan(reps)):
        # find the indices of nan neurons
        nan_ind = np.isnan(reps)
        _, _, nan_neu_ind = np.nonzero(nan_ind)

        corrected_reps = reps
        for n in np.unique(nan_neu_ind):
            # create a mask of all nan values for a neuron
            mask = np.zeros(shape=nan_ind.shape, dtype=bool)
            mask[:, :, n] = True
            masked_nan_ind = nan_ind & mask

            # substitue all nan values of neuron by average neuron response
            av_neuron_act = np.nanmean(reps[:, :, n])
            corrected_reps[masked_nan_ind] = av_neuron_act
        return corrected_reps
    else:
        return reps


def project_reps(input_reps, W_mat):
    """Project each rep of neural data using the projection matrix
    input_reps = [n_reps, n_samples, n_neurons]"""
    input_reps = fix_nan_reps(input_reps)
    reps = []
    for rep in input_reps:
        reps.append(scale(rep))
    comp_reps = np.tensordot(reps, W_mat, axes=1)
    return comp_reps


def fit_reg(X, Y):
    """
    Fits a linear regression model to the data and returns regression model with score and predictions"""
    reg = LinearRegression()
    reg.fit(X, Y)
    preds = reg.predict(X)
    score = pearsonr(Y, preds)
    return reg, score, preds


def predict_outputs(features, weights):
    """
    Predict outputs given input features and weights."""
    model_pcs = np.matmul(features - weights['pca_b'], weights['pca_w'])
    preds = np.matmul(model_pcs, weights['pls_w']) + weights['pls_b']
    return preds


def resize_mat(mat, new_size):
    """
    Resize a matrix to the desired size. Input size is [num_channels, num_pixels, num_pixels]"""
    if mat.ndim == 2:
        mat = np.expand_dims(mat, axis=0)
    num_ch, _, num_pix = np.array(mat).shape

    x = np.arange(0, num_pix)
    y = np.arange(0, num_pix)
    ratio = (new_size - 1.) / (num_pix - 1)

    x_new = np.arange(0, new_size)
    y_new = np.arange(0, new_size)

    output = []
    for i in range(num_ch):
        resized_rf_func = interp2d(x * ratio, y * ratio, mat[i], kind='cubic')
        tmp_out = resized_rf_func(x_new, y_new)
        output.append(tmp_out)

    return np.squeeze(output)
