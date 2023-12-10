from .moving_objects import load_moving_object

def load_data(batch_size, val_batch_size, data_root, num_workers):
    return load_moving_object(batch_size, val_batch_size, data_root, num_workers)
