import numpy as np

def reshape_to_graphcast(ds, batch_size=3):
    # Example: reshape (8,) time into (batch=2, time=4)
    t = ds.time.values
    batch = len(t) // batch_size
    ds = ds.isel(time=slice(0, batch * batch_size))  # trim
    ds = ds.assign_coords(time=np.arange(batch_size))
    ds = ds.stack(batch_time=['time']).unstack('batch_time')
    ds = ds.rename_dims({'batch_time_0': 'batch', 'batch_time_1': 'time'})
    return ds
