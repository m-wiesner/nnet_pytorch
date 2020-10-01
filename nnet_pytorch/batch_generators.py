# Batch generators for training and inference

def batches(dataset,n):
    for b in range(n):
        yield dataset.minibatch()

def multiset_batches(sets, genfun, *args):
    '''
        Alternating round-robin batches.
    '''
    set_batches = []
    for s in sets:
        set_batches.append(genfun(s, *args))

    if args:
        for i in range(args[0]):
            for set_batch in set_batches:
                b = next(set_batch, None)
                if b is not None:
                    yield b


def evaluation_batches(dataset):
    return dataset.evaluation_batches()

