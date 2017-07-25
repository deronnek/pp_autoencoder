# This is pp\_autoencode, the protein profile autoencoder.

## Prerequisite:
[git lfs](https://git-lfs.github.com/)

## To install:
(Optional) Create and activate a virtual environment

```
python3 -m venv autoencoder_venv
source autoencoder_venv/bin/activate
```

Fetch repository and install requirements

```
git clone https://www.github.com/deronnek/pp_autoencoder
cd pp_autoencoder  
pip install -r requirements.txt
```

Unpack data files:

```
tar jxf aa_data.tar.bz2
```

## To run demo:

`python convolutionNeuralNet.py train --seq_len 104 --n_epoch 1`

or

`python convolutionNeuralNet.py gen --seq_len 104`

or

`python convolutionNeuralNet.py test --seq_len 696`


## For command-line argument help:
`python convolutionNeuralNet.py -h`

## Troubleshooting:
If you get this message: 

`ValueError: Error when checking model target: expected convolution2d_7 to have shape (None, 8, 104, 1) but got array with shape (5022, 1, 104, 24)`

Set `image_dim_ordering` in ~/.keras/keras.json to `"th"` (NOT `"tf"`)
