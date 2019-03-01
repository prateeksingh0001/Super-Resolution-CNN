# Super-Resolution-CNN
A tensorflow implementation of the Super Resolution CNN

## Usage
Download and extract the Set5 image dataset here.

Create two folders name `Checkpoints` and `Output` in the project folder with the command 
`mkdir Checkpoints Output`

One can use this directly by importing the files `srcnn.py` and `dataset.py`

Incase one implements their own classes for pre-processing the dataset, one needs to implement the `get_next_batch(self, batch_size)` which is essentially a generator that is used by the `srcnn` model.


