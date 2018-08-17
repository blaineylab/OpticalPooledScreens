import inspect
import functools
import os
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd
import skimage
import ops.features
import ops.process
import ops.io
import ops.in_situ
from ops.process import Align


def load_well_tile_list(filename):
    wells, tiles = pd.read_pickle(filename)[['well', 'tile']].values.T
    return wells, tiles


def load_csv(f):
    with open(f, 'r') as fh:
        txt = fh.readline()
    sep = ',' if ',' in txt else '\s+'
    return pd.read_csv(f, sep=sep)


def load_pkl(f):
    return pd.read_pickle(f)


def load_tif(f):
    return ops.io.read_stack(f)


def save_csv(f, df):
    df.to_csv(f, index=None)


def save_pkl(f, df):
    df.to_pickle(f)


def save_tif(f, data_, **kwargs):
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    # make sure `data` doesn't come from the Snake method since it's an
    # argument name for the save function, too
    kwargs['data'] = data_
    ops.io.save_stack(f, **kwargs)


def restrict_kwargs(kwargs, f):
    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keep, discard = {}, {}
    for key in kwargs.keys():
        if key in f_kwargs:
            keep[key] = kwargs[key]
        else:
            discard[key] = kwargs[key]
    return keep, discard


def load_file(f):
    if not isinstance(f, str):
        raise TypeError
    if not os.path.isfile(f):
        raise IOError(2, 'Not a file: {0}'.format(f))
    if f.endswith('.tif'):
        return load_tif(f)
    elif f.endswith('.pkl'):
        return load_pkl(f)
    elif f.endswith('.csv'):
        return load_csv(f)
    else:
        raise IOError(f)


def load_arg(x):
    """What to do if load_file finds a file but raises an error?
    """
    one_file = load_file
    many_files = lambda x: [load_file(f) for f in x]
    
    for f in one_file, many_files:
        try:
            return f(x)
        except (pd.errors.EmptyDataError, TypeError, IOError) as e:
            if isinstance(e, (TypeError, IOError)):
                # wasn't a file, probably a string arg
                pass
            elif isinstance(e, pd.errors.EmptyDataError):
                # failed to load file
                return None
            pass
    else:
        return x


def save_output(f, x, **kwargs):
    """Saves a single output file. Can extend to list if needed.
    Saving .tif might use kwargs (luts, ...) from input.
    """
    if x is None:
        # need to save dummy output to satisfy Snakemake
        with open(f, 'w') as fh:
            pass
        return
    f = str(f)
    if f.endswith('.tif'):
        return save_tif(f, x, **kwargs)
    elif f.endswith('.pkl'):
        return save_pkl(f, x)
    elif f.endswith('.csv'):
        return save_csv(f, x)
    else:
        raise ValueError('not a recognized filetype: ' + f)


def get_arg_names(f):
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]


def get_kwarg_defaults(f):
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return {}
    defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults


def remove_channels(data, remove_index):
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    channels_mask[remove_index] = False
    data = data[..., channels_mask, :, :]
    return data


def call_from_snakemake(f):
    """Turn a function that acts on a mix of image data, table data and other 
    arguments and may return image or table data into a function that acts on 
    filenames for image and table data, plus other arguments.

    If output filename is provided, saves return value of function.

    Supported filetypes are .pkl, .csv, and .tif.
    """
    def g(**kwargs):

        # split keyword arguments into input (needed for function)
        # and output (needed to save result)
        input_kwargs, output_kwargs = restrict_kwargs(kwargs, f)

        # load arguments provided as filenames
        input_kwargs = {k: load_arg(v) for k,v in input_kwargs.items()}

        result = f(**input_kwargs)

        if 'output' in output_kwargs:
            save_output(output_kwargs['output'], result, **output_kwargs)

    return functools.update_wrapper(g, f)


class Snake():
    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], call_from_snakemake(f))

    @staticmethod
    def _align(data, method='DAPI', upsample_factor=2, window=4):
        """Expects input array of dimensions (CYCLE, CHANNEL, I, J).
        If window is 
        """
        data = np.array(data)
        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # align between SBS channels for each cycle
        aligned = data.copy()
        align_it = lambda x: Align.align_within_cycle(x, window=window, upsample_factor=upsample_factor)
        aligned[:, 1:] = np.array([align_it(x) for x in aligned[:, 1:]])

        if method == 'DAPI':
            # align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=0, 
                                window=window, upsample_factor=upsample_factor)
        elif method == 'SBS_mean':
            # calculate cycle offsets using the average of SBS channels
            target = Align.apply_window(aligned[:, 1:], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target)
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

        return aligned

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.
        Use `skip_index` to skip a channel (e.g., DAPI with `skip_index=0`).
        """
        data = np.array(data)
        loged = ops.process.log_ndi(data, sigma=sigma)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation to estimate sequencing read locations.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        leading_dims = tuple(range(0, data.ndim - 2))
        consensus = np.std(data, axis=leading_dims)

        return consensus
    
    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max):
        """Find nuclei from DAPI. Find cell foreground from aligned but unfiltered 
        data. Expects data to have shape C x I x J.
        """

        if isinstance(data, list):
            dapi = data[0]
        elif data.ndim == 3:
            dapi = data[0]

        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_cells(data, nuclei, threshold):
        """Segment cells from aligned data. To use less than full cycles for 
        segmentation, filter the input files. Matches cell labels to nuclei labels.
        """
        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, 1:].min(axis=0).mean(axis=0)
        else:
            mask = np.median(data[1:], axis=0)

        mask = mask > threshold
        try:
            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cells = ops.process.find_cells(nuclei, mask)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        return cells

    @staticmethod
    def _find_peaks(data):
        if data.ndim == 2:
            data = [data]
        peaks = [ops.process.find_peaks(x) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks).squeeze()
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        import scipy.ndimage.filters

        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = remove_channels(data, remove_index)
        
        maxed = np.zeros_like(data)
        maxed[:, 1:] = scipy.ndimage.filters.maximum_filter(data[:,1:], size=(1, 1, width, width))
        maxed[:, 0] = data[:, 0]  # DAPI
    
        return maxed

    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_std, wildcards, bases='GTAC'):
        """Assumes sequencing covers 'GTAC'[:channels].
        """

        if maxed.ndim == 3:
            maxed = maxed[None]

        if len(bases) != maxed.shape[1]:
            error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        values, labels, positions = (
            ops.in_situ.extract_base_intensity(maxed, peaks, cells, threshold_std))

        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        for k,v in wildcards.items():
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(df_bases):
        """Median correction performed independently for each tile.
        """
        if df_bases is None:
            return
        
        cycles = len(set(df_bases['cycle']))
        return (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call, cycles)
            )

    @staticmethod
    def _call_cells(df_reads):
        """Median correction performed independently for each tile.
        """
        if df_reads is None:
            return
        
        return ops.in_situ.call_cells(df_reads)

    @staticmethod
    def _align_phenotype(data_DO, data_phenotype):
        """Align using DAPI.
        """
        _, offset = ops.process.Align.calculate_offsets([data_DO[0], data_phenotype[0]])
        offsets = [offset] * len(data_phenotype)
        aligned = ops.process.Align.apply_offsets(data_phenotype, offsets)
        return aligned

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from ops.features import features_frameshift
        return Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift)       

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, data_sbs_1, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from ops.features import features_frameshift_myc
        return Snake._extract_features(data_phenotype, nuclei, wildcards, features)     

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        import ops.features

        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        features_n = {k + '_nuclear': v for k,v in features_n.items()}
        features_c = {k + '_cell': v    for k,v in features_c.items()}

        df_n =  Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
        df_c =  Snake._extract_features(data_phenotype, cells, wildcards, features_c) 

        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('cell'), df_c.set_index('cell')], axis=1, join='inner')
                .reset_index())

        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)

    @staticmethod
    def _extract_features(data, nuclei, wildcards, features=None):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        from ops.process import feature_table
        from ops.features import features_cell
        features = features.copy() if features else dict()
        features.update(features_cell)

        df = feature_table(data, nuclei, features)

        for k,v in wildcards.items():
            df[k] = v
        
        return df

    @staticmethod
    def _extract_minimal_phenotype(data_phenotype, nuclei, wildcards):
        return Snake._extract_features(data, nuclei, wildcards, dict())


Snake.load_methods()