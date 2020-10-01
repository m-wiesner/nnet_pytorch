# Batch generators for training and inference

def batches(dataset, n):
    for b in range(n):
        yield dataset.minibatch()

def multiset_batches(sets, genfun, *args):
    '''
        Alternating round-robin batches.
    '''
    # We assume the generators are of equal length
    for set_batches_n in zip(*[genfun(s, *args) for s in sets]):
        for b in set_batches_n:
            if b is not None:
                yield b
    
def evaluation_batches(dataset):
    return dataset.evaluation_batches()

