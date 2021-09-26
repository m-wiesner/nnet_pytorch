# Copyright 2021
# Apache 2.0

from collections import namedtuple


class NnetPytorchDataset(object):

    Minibatch = namedtuple('Minibatch', ['input', 'target', 'metadata'])

    @staticmethod
    def add_args(parser):
        pass
    
    @classmethod
    def build_dataset(cls, args):
        raise NotImplementedError
       
    def __init__(self):
        pass


    def __len__(self):
        '''
            Returns the total number of elements in the dataset
        '''
        raise NotImplementedError

    def size(self, idx):
        '''
            Returns some notion of size of an individual element
            of the dataset.
        '''
        raise NotImplementedError(
            "This function should return the size of an individual element of "
            "the dataset."
        )

    
    def minibatch(self):     
        '''
            This is effectively the collater. It defines how multiple elements
            of a dataset are aggregated or collated together for neural network
            training.
        '''
        raise NotImplementedError(
            "This function should return an object that groups together "
            "different elements of the dataset for neural network training."
        )

    def evaluation_batches(self):
        '''
            This yields batches of the evaluation set.
        '''
        raise NotImplementedError(
            "This function should yield batches of the eval data."
        )


    def __getitem__(self, idx):
        raise NotImplementedError(
            "This function is used to return a formatted inputs and outputs "
            "for a single element from the dataset. self.minibatch() "
            "should make repeated calls to __getitem__ when forming "
            "minibatches. The argument idx can be any hashable object."
        )


    def move_to(self, b, device):
        pass

