import numpy as np

def aa_key(aa_vec):
    """ This function allows us to sort the rows of a one-hot PSSM with 22
    values such that they are grouped by chemical properties. 

    Parameters
    ----------
    aa_vec: numpy array
        The row of the PSSM for which we need a key to order by

    Returns
    -------
    The sort key for the input row
    
    """
    aa_char = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
    go      = list('RHKDESTNQCGPAVILMFYWX')
    go.append('NoSeq')
    char    = aa_char[np.argmax(aa_vec[:22])]
    return go.index(char)
 
def order_pssm_rows_by_aa_type(mat):
    """ This function sorts the rows of an input one-hot PSSM such that they
    are grouped by chemical properties.

    Parameters
    ----------
    mat: numpy matrix
        one-hot PSSM with 22 amino acid types to be sorted

    Returns
    -------
    A sorted copy of the matrix

    """
    ret = np.zeros(mat.shape)
    for seq in range(mat.shape[0]):
        tmp = mat[seq].tolist()
        x   = sorted(tmp, key=aa_key)
        ret[seq] = np.array(x)
    return ret

def reorder_pssm_cols(mat, pssm_offset=22, original_order=None, grouped_order=None, w=0):
    """ This function sorts the columns of a pssm such that they are grouped by
    chemical properties.  Note that no signal is lost in this operation, unlike
    in order_pssm_rows_by_aa_type.

    Parameters
    ----------
    mat: numpy matrix
        The matrix to be reordered.
        mat must be a 3-dimensional numpy array, with the last dimension being the
        sequence+pssm, where the pssm starts at index pssm_offset

    pssm_offset: integer
        The index into mat at which the pssm begins.
        Default: 22

    original_order: string
        The order of the amino acids in the incoming mat.
        Default: 'ACDEFGHIKLMNPQRSTVWXY'

    grouped_order: string
        The desired order of the amino acids 
        Default: 'RHKDESTNQCGPAVILMFYWX'

    w: integer
        The number of amino acids on either side of a center amino acid
        included in mat.
        Default: 0 

    """

    # w amino acids on either side of center, plus center.  W=0 means just the center,
    # which we still need a column for (so w is always at least 1)
    w = w*2+1
    ret = np.zeros(mat.shape)
    if original_order == None:
        oo = list('ACDEFGHIKLMNPQRSTVWXY')
    else:
        oo = original_order

    # grouping a.a.s by type, we have: RHK DE STNQ CUGP AVILMF YW
    if grouped_order == None:
        go = list('RHKDESTNQCGPAVILMFYWX')
        # Repeat each character individually w times (see comment above about
        # value of w)
        go = [x for a in go for x in list(a*w)]
        # This is equivalent to:
        # res = []
        # for a in go:
        #   for x in list(a*w):
        #       res.append(x)
        # go = res
    else:
        go = grouped_order

    # Loop over columns and find sources to copy from
    for i in range(len(go)):
        # w is accounted for in how we construct go
        source_col = oo.index(go[i])
        source_col = source_col + (len(go) * (i // len(go))) 
        
        ret[:,:,i+pssm_offset] = mat[:,:,source_col+pssm_offset]

    # Slot in the one-hot portion into the return matrix as well
    ret[:,:,:pssm_offset] = mat[:,:,:pssm_offset]

    return ret

def scale_pssm_by_mean_and_std(mat, pssm_offset=22):
    """ This function converts the values of a pssm into z-scores.
    Parameters
    ----------
    mat: 3-d numpy matrix
        mat must be a 3-dimensional numpy array, with the last dimension being the
        sequence+pssm, where the pssm starts at index pssm_offset

    pssm_offset: integer
        The index into mat at which the pssm begins.
        Default: 22

    Returns
    -------
    A copy of mat with the pssm values scaled

    """
    from scipy import stats
    ret = np.zeros(mat.shape)
    for seq in range(mat.shape[0]):
        ret[seq, :, pssm_offset:]  = stats.zscore(mat[seq, :, pssm_offset:], axis=1)
        for aa in range(ret.shape[1]):
            ret[seq, aa, pssm_offset:]  = stats.zscore(mat[seq, aa, pssm_offset:])
    return ret

def fill_no_seq_X_with_repeats(mat):
    """ This function converts NoSeq and X entries at the end of a pssm into
    copies of the information from the beginning of the sequence.  This is done
    so that all input sequences to a neural network have (roughly) the same
    information density.

    Parameters
    ----------
    mat: 3-d numpy matrix
        mat must be a 3-dimensional numpy array, with the last dimension being the
        sequence+pssm, where the pssm starts at index pssm_offset

    Returns
    -------
    A copy of mat with the NoSeq and X entries replaced 

    """

    empty_vec = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0., 0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 1.]])
    ret       = np.zeros(mat.shape)
    for seq in range(mat.shape[0]):
        copied = 0
        for aa in range(mat.shape[1]):
            if (mat[seq, aa, :len(empty_vec)] == empty_vec).all():
                # Reset if we've started copying from the empty stuff
                if (mat[seq, copied, :len(empty_vec)] == empty_vec).all(): 
                    copied = 0
                ret[seq, aa] = mat[seq, copied]
                copied += 1
            else:
                ret[seq, aa] = mat[seq, aa]
    return ret

def wmers_as_columns(df, w):
    """ This function takes w rows from either side of a center row and
    concatenates them as additional columns.
    Border cases
    ------------
    At the beginning: all rows indexed <= w get the first w*2+1 rows as their w-mer
    At the end: all rows indexed > length - w -1  get the last w*2+1 rows as their w-mer

    Parameters
    ----------
    df: pandas dataframe 
       PSI-BLAST pssm loaded into a pandas dataframe 

    w: integer
        Number of amino acids to include around a central amino acid

    Returns
    -------
    A new dataframe with the w-mers as columns

    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise Exception("wmers_as_columns requires a pandas dataframe as the first argument, got: %s" %(type(df),))

    if len(df) < w*2+1:
        raise Exception("Not enough rows for size of window")

    ret = pd.DataFrame()
    for i in range(len(df)):
        if i <= w:
            # Beginning border case (doesn't depend on i)
            wmer = pd.DataFrame(df[0 : w*2 + 1].values.reshape(1, -1))
        elif i > len(df) - w - 1:
            # Ending border case (doesn't depend on i)
            wmer = pd.DataFrame(df[len(df) - w*2 - 1 : len(df)].values.reshape(1, -1))
        else:
            # All middle cases
            wmer = pd.DataFrame(df[i-w : i+w+1].values.reshape(1, -1))
        ret = ret.append(wmer)
    return ret


def pssm_to_wmer_pssm(mat, w=0, pssm_offset=22):
    """ This function is a utility function to take a pssm in the form of a 3-d
    numpy matrix and incorporate w-mer information.  All the heavy lifting is
    done by wmers_as_columns
    Parameters
    ----------
    mat: 3-d numpy matrix
        mat must be a 3-dimensional numpy array, with the last dimension being the
        sequence+pssm, where the pssm starts at index pssm_offset

    pssm_offset: integer
        The index into mat at which the pssm begins.
        Default: 22

    w: integer
        Number of amino acids to include around a central amino acid
        Default: 0
    """

    import pandas as pd
    # Convert to wmers
    wmat = []
    for p in range(mat.shape[0]):
        pssm = wmers_as_columns(pd.DataFrame(mat[p, :, pssm_offset:]), w)
        wmat.append(pssm.values)
    wmat = np.array(wmat)

    # Store back in mat numpy matrix
    mat = np.concatenate([mat[:, :, :pssm_offset], wmat], axis=2)
    return mat

def pssm_to_image_representation(seq_len, min_dim, max_dim, matrix):
    """ This function re-shapes a matrix to look like an image with 1 channel
     this must be called on train, test, and valid the img_convolution model is
     to be used.

     Parameters
     ----------
     seq_len: integer
        The length of the input sequences

    min_dim: integer
        The minimum dimension to consider for each input vector

    max_dim: integer
        The maximum dimension to consider for each input vector

    matrix: numpy array
        The 3-d input matrix to be re-shaped

    Returns
    -------
    A matrix with an additional dimension added

    """

    # Adding 2 so the downsampling/upsampling are symmetric
    input_len = (max_dim-min_dim)+2

    # Create placeholder of the proper shape and with bigger input_len
    matrix_placeholder = np.zeros((len(matrix), 1, seq_len, input_len))
    
    # Reset input_len to match input matrix
    input_len = (max_dim-min_dim)

    # Reshape matrix to fit in proper slot of placeholder
    matrix   = np.reshape(matrix, (len(matrix), 1, seq_len, input_len))
    
    # Fill (most of) placeholder
    matrix_placeholder[:matrix.shape[0], :matrix.shape[1], :matrix.shape[2], :matrix.shape[3]] = matrix

    return matrix_placeholder
