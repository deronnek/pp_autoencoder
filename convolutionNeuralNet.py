import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from datetime import datetime
from sklearn.metrics import roc_auc_score
import sys
import os
import pickle
import scipy.sparse
from nnUtils import *
import argparse
import logging

# TODO: 
# 10.  Start with a fresh virtual environment, install requirements.txt and try
# to run code to make sure it works

def img_convolution(seq_len, min_dim, max_dim):
    """ This function creates a convolution neural network
    encoder/auto-encoder for use with amino acid sequences that have been
    represented as images

    Parameters
    ----------
    seq_len: integer
        The number of amino acids in each sequence

    min_dim: integer
        The first dimension to consider for input feature vectors

    max_dim: integer
        The last dimension to consider for input feature vectors

    Returns
    -------
    (encoder, autoencoder)
    
    Where encoder is a keras Model for use in creating feature vectors, and
    autoencoder is a keras Model for use in training and validating the Model

    """
        
    input_len = (max_dim-min_dim)+2
    input_img = Input(shape=(1, seq_len, input_len))

    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    
    encoder = Model(input_img, encoded)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 9, 9, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 9, 9, activation='relu', border_mode='same')(x)

    autoencoder = Model(input_img, decoded)
    # These are useful when making network design alterations
    #autoencoder.summary()
    #encoder.summary()
    autoencoder.compile(optimizer='adadelta', loss='mse')

    return (encoder, autoencoder)

def preprocess_matrix(mat, seq_off, seq_len, w=0, to_wmer=True, pssm_offset=22):
    """Perform transformations on pssms to purify signal.  Reorder columns, rows, 
    convert to wmers and mean-center/scale. This should be run on all matricies
    before handing them to the neural network model
   
    Parameters
    ----------
    mat: numpy matrix
        matrix to be processed

    seq_off: integer
        offset into sequences, used in preprocessing matrix

    seq_len: integer
        sequence length, used in preprocessing matrix

    w: integer
        length of w-mers to use if to_wmer is True

    to_wmer: Bool
        whether or not to convert to w-mers

    Returns
    -------
    Preprocessed matrix in numpy format

    """

    mat = mat[:, seq_off:seq_len, :]

    if w > 0 and to_wmer:
        mat = pssm_to_wmer_pssm(mat, w=w, pssm_offset=pssm_offset)
        
    mat = fill_no_seq_X_with_repeats(mat)
    logger.debug("Fill done")
    mat = scale_pssm_by_mean_and_std(mat)
    logger.debug("Scale done")
    mat = order_pssm_rows_by_aa_type(mat)
    logger.debug("Reorder rows done")
    mat = reorder_pssm_cols(mat, w=w)
    logger.debug("Reorder cols done")
    return mat

def read_recreate_derived_matrix(mat, derived_matrix_file, seq_off, seq_len, w,
                                 to_wmer=True):
    """Attempt to read a matrix from derived_matrix_file, and if unsuccessful,
    recreate that matrix using preprocess_matrix.  If we didn't put a '.npy' at the
    end, tries to load the file with that extension as well before
    regenerating.
   
    Parameters
    ----------
    mat: numpy matrix
        (raw) unprocessed matrix, used if recreating

    derived_matrix_file: string
        filename of preprocessed matrix in numpy format, read if it exists,
        *written* if it does not

    seq_off: integer
        offset into sequences, used in preprocessing matrix

    seq_len: integer
        sequence length, used in preprocessing matrix

    w: integer
        length of w-mers to use if to_wmer is True

    to_wmer: Bool
        whether or not to convert to w-mers

    Returns
    -------
    derived matrix in numpy format

    Note:
        This function also saves the preprocessed matrix to derived_matrix_file
    """

    try:
        dmat = np.load(derived_matrix_file)
        logger.debug("Derived matrix loaded from %s" %(derived_matrix_file,))
    except OSError:
        try:
            dmat = np.load(derived_matrix_file+'.npy')
            logger.debug("Derived matrix loaded from %s" %(derived_matrix_file+'.npy',))
        except OSError:
            logger.debug("Re-ordering matrix and saving")
            dmat = preprocess_matrix(mat, seq_off, seq_len, w, to_wmer=to_wmer)
            np.save(derived_matrix_file, dmat)
            logger.debug("Wrote: %s" %(derived_matrix_file,))
    logger.debug("Derived matrix shape:",dmat.shape)
    return dmat

def train_model_with_callbacks(model, x_train, x_valid, nb_epoch, seqlen, wmer, checkpoint_path='./data'):
    """Wrapper around the keras model.fit call to add callbacks for monitoring, early
    stopping, and checkpointing along the way.  Basically the call to model.fit
    has too many parameters that I never change but don't want the defaults
    for, and looks ugly.  Also I had it in more than one place at some point
    
    Parameters
    ----------
    model: keras model object
        Model to be fit to data

    x_train: numpy array
        Training data

    x_valid: numpy array
        Validation data

    nb_epoch: integer
        Number of epochs to train for

    seqlen: integer
        The length of the input sequences

    wmer: integer
        The width of the w-mer considered

    checkpoint_path: string
        Path in which to store checkpoint files
        Default: .

    Returns
    -------
    Nothing

    Side-effects
    ------------
    Model is trained (just an effect, really)

    """

    checkpointFilePath =\
    '%s/seqlen.%d.wmer.%d.weights.{epoch:02d}-{val_loss:.2f}.hdf5' %(checkpoint_path, seqlen, wmer)

    model.fit(x_train, x_train, nb_epoch=nb_epoch, batch_size=128, shuffle=True, validation_data=(x_valid, x_valid),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder/'+datetime.now().strftime("%Y%m%d-%H%M%S")+"/"),
                            EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=0, mode='auto'),
                            ModelCheckpoint(checkpointFilePath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)])

def train_network(args, autoencoder):
    """Train a convolution neural network based on command line parameters.
    Builds the input matricies and calls model.fit, reporting results to
    tensorboard.

        Parameters
        ----------
        args: argparse object
            Command-line options

        autoencoder: keras model object
            Model to be trained

        Returns
        -------
        Nothing
    """
            
    mat     = np.load(args.no_label_file)
    seq_off = args.seq_off
    seq_len = args.seq_len
    w       = args.wsize
    min_dim = args.min_dim
    max_dim = args.max_dim

    split_seq_index = args.split_seq_index
    end_seq_index   = args.end_seq_index
    logger.debug("Initial matrix loaded")
    mat     = read_recreate_derived_matrix(mat, args.reordered_file+'.%d.wmer.%d'
                                           %(seq_len,w), seq_off, seq_len, w)

    x_train = mat[:split_seq_index,       seq_off:seq_len, min_dim:max_dim]
    x_valid = mat[split_seq_index:end_seq_index, seq_off:seq_len, min_dim:max_dim]
    x_test  = mat[end_seq_index:,         seq_off:seq_len, min_dim:max_dim]

    x_train = pssm_to_image_representation(seq_len, min_dim, max_dim, x_train)
    x_test  = pssm_to_image_representation(seq_len, min_dim, max_dim, x_test)
    x_valid = pssm_to_image_representation(seq_len, min_dim, max_dim, x_valid)

    train_model_with_callbacks(autoencoder, x_train, x_valid, args.n_epoch, seq_len, w)
    autoencoder.save_weights(args.weights)

def test_network(args, autoencoder, encoder):
        """
        This function:
            1.  Reads in files from ./data (pssms and family labels) 
            2.  Generates new re-arranged matricies from them and vectors for each sequence
            3.  Predicts families, and then computes AUC for ROC and ROC50 
            
            The basic idea being that if the encoded vectors are good, they
            structurally similar proteins should have similar vectors.

        Parameters
        ----------

        args: argparse object
            Command-line options

        autoencoder: Keras model object
            Pre-trained autoencoder network

        encoder: Keras model object
            Pre-trained encoder network

        Returns
        -------
        Nothing, but outputs the test results to the log

        """
        weights_file           = args.weights
        w                      = args.wsize
        seq_len                = args.seq_len
        seq_off                = args.seq_off
        min_dim                = args.min_dim
        max_dim                = args.max_dim

        test_mat_file          = './data/nnFormat.wmer.%d.pssms.npy' %(w,)
        row_labels_file        = './data/nnFormat.wmer.%d.rowLabels' %(w,)
        family_membership_file = './data/all_seq_fam_membership.txt'
        processed_matrix_file  = './data/clanrows.%d.wmer.%d' %(seq_len, w)

        logger.debug("Reading SCOP data files")
        test_mat               = np.load(test_mat_file)
        row_index_for_seq_id   = pickle.load(open(row_labels_file, 'rb'))
        logger.debug("test_mat shape:",test_mat.shape)

        # Read the family label file into a pandas dataframe
        df                     = pd.read_csv(family_membership_file, sep=' ', index_col=0)

        # read_recreate_derived_matrix actually truncates to seq_off:seq_len 
        # These have already been turned into wmers so to_wmer=False
        test_mat     = read_recreate_derived_matrix(test_mat,
                                                    processed_matrix_file,
                                                    seq_off, seq_len, 
                                                    w, to_wmer=False)

        test_mat     = test_mat[:, seq_off:seq_len, min_dim:max_dim]
        test_mat     = pssm_to_image_representation(seq_len, min_dim, max_dim,
                                                    test_mat)

        logger.info("SCOP files read")

        autoencoder.load_weights(weights_file)
        logger.info("Autoencoder weights loaded")
        
        try:
            pred_vectors_file = args.weights+'.predictedVectors.wmer.%d.npy' %(w,)
            pred_vectors      = np.load(pred_vectors_file)
            logger.info("Predicted vectors read from %s" %(pred_vectors_file,))
        except FileNotFoundError:
            logger.info("Predicting vectors...")
            pred_vectors = []
            for i in range(len(test_mat)):
                x = encoder.predict(test_mat[i:i+1])
                pred_vectors.append(x.flatten())
            np.save(pred_vectors_file, np.array(pred_vectors))
            logger.info("Predicted vectors saved to %s" %(pred_vectors_file,))


        # down each column of the label matrix, build a centroid vector for this family
        # from the positive training examples
        centroids  = {}
        for family in df.columns:
            first       = True
            centroid    = []
            n_positives = 0
            for seq_id in df[family].index:
                if df[family][seq_id] == 1: # 1 = positive train
                    if first:
                        centroid = pred_vectors[row_index_for_seq_id[seq_id]]
                        first    = False
                    else:
                        # add values to existing centroid
                        centroid = np.sum((centroid,
                                           pred_vectors[row_index_for_seq_id[seq_id]]), axis=0)
                    n_positives += 1

            if centroid != []:
                centroid = centroid / n_positives
                centroids[family] = centroid 

        # down each column of the label matrix, build a label vector and prediction
        # vector
        logger.info("Family AUC")
        auc_results = []
        for family in df.columns:
            label_vec = []
            pred_vec  = []
            for seq_id in df[family].index:
                if not family in centroids:
                    break
                # negative test or positive test; 3/4 were assigned by SCOP
                if df[family][seq_id] == 3 or df[family][seq_id] == 4: 
                    # Get prediction from neural net via dot product to family --
                    # normalize by length to get cosine similarity.  
                    # TODO: benchmark this against sklearn.metrics.cosine_similairty
                    pred_val = np.dot(pred_vectors[row_index_for_seq_id[seq_id]], centroids[family])
                    pred_val /= np.linalg.norm(centroids[family], ord=2) *\
                        np.linalg.norm(pred_vectors[row_index_for_seq_id[seq_id]], ord=2)
                    pred_vec.append(pred_val)
                    if df[family][seq_id] == 3:
                        label_vec.append(1)
                    if df[family][seq_id] == 4:
                        label_vec.append(0)

            auc_score = roc_auc_score(label_vec, pred_vec)
            auc_results.append(auc_score)
            logger.info("%s %f" %(family, auc_score))
        logger.info("Mean AUC: %f" %(np.mean(np.array(auc_results))))

def generate_vectors(args, encoder):
    w       = args.wsize
    seq_len = args.seq_len
    seq_off = args.seq_off
    min_dim = args.min_dim
    max_dim = args.max_dim

    weights_file     = args.weights
    pssm_numpy_file  = args.numpy_pssm
    test_mat         = np.load(pssm_numpy_file)
    baseDir          = os.path.dirname(pssm_numpy_file)
    logger.debug("test_mat shape:",test_mat.shape)

    # read_recreate_derived_matrix truncates to seq_off:seq_len 
    # These have already been turned into wmers so to_wmer=False
    test_mat     = read_recreate_derived_matrix(test_mat,
                                               baseDir+'/derived.nnFormat.%d.wmer.%d'
                                               %(seq_len, w), seq_off, seq_len,
                                                w, to_wmer=False)
    test_mat     = test_mat[:, seq_off:seq_len, min_dim:max_dim]
    test_mat     = pssm_to_image_representation(seq_len, min_dim, max_dim,
                                                test_mat)

    logger.debug("Loading Autoencoder weights from %s" %(weights_file,))
    autoencoder.load_weights(weights_file)
    logger.debug("Autoencoder weights loaded from %s" %(weights_file,))
    
    logger.info("Predicting vectors...")
    # This should output a scipy sparse matrix with one vector per row (even though 
    # they're dense at this point)
    pred_vectors_file  = args.outfile
    pred_vectors = []
    for i in range(len(test_mat)):
        x = encoder.predict(test_mat[i:i+1])
        pred_vectors.append(x.flatten())
    pred_vectors = np.array(pred_vectors)
    logger.debug("Pred vectors shape: ",pred_vectors.shape)
    sparse_pred_vectors = scipy.sparse.csr_matrix(pred_vectors)
    pickle.dump(sparse_pred_vectors, open(pred_vectors_file, 'wb'))
    logger.info("Predicted vectors saved to %s with shape %s"
                %(pred_vectors_file, sparse_pred_vectors.shape))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=['train', 'test', 'gen'], help="Mode of\
                        operation:\n\ttrain: train a model\n\ttest: test a\
                        model\n\tgen: generate vectors for prediction")
    parser.add_argument("--numpy_pssm", default='./data/nnFormat.wmer.0.pssms.npy', help="Numpy matrix of pssm data created by psiBlastPSSMToNNFormat.py (used with mode gen)") 
    parser.add_argument("--wsize", type=int, default=0, help="Number of amino acids on either side of center to consider")
    parser.add_argument("--resume_file", help="Weights file to resume training from")
    parser.add_argument("--weights", default='./data/model.weights', help="Weights file to use in testing/generating")
    parser.add_argument("--n_epoch", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--seq_off", type=int, default=0, help="Start this far into sequences")
    parser.add_argument("--seq_len", type=int, default=696, help="Sequence length (and width of input layer)")
    parser.add_argument("--end_seq_index", type=int, default=5278, help="The first index of the testing set")
    parser.add_argument("--split_seq_index", type=int, default=5022, help="The first index of the validation set")
    parser.add_argument("--min_dim", type=int, default=22, help="Minimum dimension in feature vector.  0 for sequence, 22 for profile")
    parser.add_argument("--max_dim", type=int, default=44, help="Maximum dimension in feature vector.  20 for sequence, 44 for profile, 43 if you want to skip the 'NoSeq' column")
    parser.add_argument("--outfile", default='./generated.vectors', help="Feature vector outfile to use with mode gen")
    parser.add_argument("--no_label_file",
                        default='./data/cullpdb+profile_6133_filtered.KDR.nolabels.npy',
                        help="The original pssm matrix file to preprocess and use in training.")
    parser.add_argument("--reordered_file",
                        default='./data/cullpdb+profile_6133_filtered.KDR.nolabels.reOrderedRowsAndCols.npy',
                        help="The preprocessed pssm matrix file to use in training.")

    #--------------------------------------------------------------------------------
    # Preliminary set up and argument parsing
    #--------------------------------------------------------------------------------
    args    = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger(__name__)

    w = args.wsize
    if w > 0:
        args.max_dim = args.min_dim*(w*2+1)+args.min_dim
        logger.info("Using wmer rows with w=%d; max_dim: %d" %(w, args.max_dim))

    # We need the network architecture regardless of whether we're training, generating vectors, or testing
    encoder, autoencoder = img_convolution(args.seq_len, args.min_dim,
                                           args.max_dim)

    if args.resume_file is not None:
        autoencoder.load_weights(args.resume_file)
        logger.info("Weights loaded from %s" %(args.resume_file,))

    #--------------------------------------------------------------------------------
    # Main mode-switching block 
    #--------------------------------------------------------------------------------
    if args.mode == 'train':
        train_network(args, autoencoder)
    elif args.mode =='gen':
        generate_vectors(args, encoder)
    elif args.mode =='test':
        test_network(args, autoencoder, encoder)
