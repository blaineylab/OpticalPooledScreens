import numpy as np
import pandas as pd
from scipy.spatial.kdtree import KDTree
from scipy.stats import mode
from pims import ND2_Reader

from ops.constants import *

def add_neighbors(df_info, num_neighbors=9, radius_leniency=10):
    xy = ['x_um', 'y_um']
    xy = [GLOBAL_X, GLOBAL_Y]
    site = SITE
    
    df = df_info.drop_duplicates(xy)
    kdt = KDTree(df[xy].values)

    distance, index = kdt.query(df[xy].values, k=num_neighbors)

    # convert to m
    index = np.array(df[site])[index]

    m = mode(distance.max(axis=1).astype(int)).mode[0]
    filt = (distance < (m + radius_leniency)).all(axis=1)

    it = zip(df.loc[filt, site], index[filt])
    arr = []
    for m, ix in it:
        arr += [{site: m, 'ix': sorted(ix)}]

    return df_info.merge(pd.DataFrame(arr), how='left')


def get_metadata_at_coords(nd2, **coords):
    import pims_nd2
    h = pims_nd2.ND2SDK

    _coords = {'t': 0, 'c': 0, 'z': 0, 'o': 0, 'm': 0}
    _coords.update(coords)
    c_coords = h.LIMUINT_4(int(_coords['t']), int(_coords['m']), 
                           int(_coords['z']), int(_coords['o']))
    i = h.Lim_GetSeqIndexFromCoords(nd2._lim_experiment,
                                c_coords)

    h.Lim_FileGetImageData(nd2._handle, i, 
                           nd2._buf_p, nd2._buf_md)


    return {'x_um': nd2._buf_md.dXPos,
                    'y_um': nd2._buf_md.dYPos,
                    'z_um': nd2._buf_md.dZPos,
                    't_ms': nd2._buf_md.dTimeMSec,
                }    

def extract_nd2_metadata(f, interpolate=True, progress=None):
    """Interpolation fills in timestamps linearly for each well; x,y,z positions 
    are copied from the first time point. 
    """
    nd2 = ND2_Reader(f)

    ts = range(nd2.sizes['t'])
    ms = range(nd2.sizes['m'])   

    if progress is None:
        progress = lambda x: x

    arr = []
    for t, m in progress(list(product(ts, ms))):
        boundaries = [0, nd2.sizes['m'] - 1]
        skip = m not in boundaries and t > 0
        if interpolate and skip:
            metadata = {}
        else:
            metadata = get_metadata_at_coords(nd2, t=t, m=m)
        metadata['t'] = t
        metadata['m'] = m
        metadata['file'] = f
        arr += [metadata]
    
    nd2.close()
        
    df_info = pd.DataFrame(arr)
    if interpolate:
        return (df_info
         .sort_values(['m', 't'])
         .assign(x_um=lambda x: x['x_um'].fillna(method='ffill'))
         .assign(y_um=lambda x: x['y_um'].fillna(method='ffill'))        
         .assign(z_um=lambda x: x['z_um'].fillna(method='ffill'))         
         .sort_values(['t', 'm'])
         .assign(t_ms=lambda x: x['t_ms'].interpolate())
                )
    else:
        return df_info

