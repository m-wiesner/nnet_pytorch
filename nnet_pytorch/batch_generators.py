# Batch generators for training and inference

def batches(dataset, n):
    for b in range(n):
        yield dataset.minibatch()

def multiset_batches(sets, genfun, *args):
    '''
        Alternating round-robin batches.
    '''
    set_batches = []
    for s in sets:
        set_batches.append(genfun(s, *args))

    # We assume the generators are of equal length
    for set_batches_n in zip(*set_batches):
        for set_batch in set_batches_n:
            b = next(set_batch, None)
            if b is not None:
                yield b
    
def evaluation_batches(dataset):
    return dataset.evaluation_batches()

