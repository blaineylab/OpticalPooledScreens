import functools

from string import Formatter
from itertools import product
from decorator import decorator

import numpy as np
import pandas as pd


# PYTHON
class Memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    Numpy arrays are treated specially with `copy` kwarg.

    Based on http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kwargs):
        key = str(args) + str(kwargs)
        try:
            if isinstance(self.cache[key], np.ndarray):
                if kwargs.get('copy', True):
                    return self.cache[key].copy()
                else:
                    return self.cache[key]
            return self.cache[key]
        except KeyError:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        fn = functools.partial(self.__call__, obj)
        fn.reset = self._reset
        return fn

    def _reset(self):
        self.cache = {}


# PANDAS
def bin_join(xs, symbol):
    symbol = ' ' + symbol + ' ' 
    return symbol.join('(%s)' % x for x in xs)
        

or_join  = functools.partial(bin_join, symbol='|')
and_join = functools.partial(bin_join, symbol='&')


def groupby_reduce_concat(gb, *args, **kwargs):
    """
    df = (df_cells
          .groupby(['stimulant', 'gene'])['gate_NT']
          .pipe(groupby_reduce_concat, 
                fraction_gate_NT='mean', 
                cell_count='size'))
    """
    for arg in args:
        kwargs[arg] = arg
    reductions = {'mean': lambda x: x.mean(),
                  'std': lambda x: x.std(),
                  'sem': lambda x: x.sem(),
                  'size': lambda x: x.size(),
                  'count': lambda x: x.size(),
                  'sum': lambda x: x.sum(),
                  'sum_int': lambda x: x.sum().astype(int),
                  'first': lambda x: x.nth(0),
                  'second': lambda x: x.nth(1)}
    
    for arg in args:
        if arg in reductions:
            kwargs[arg] = arg

    arr = []
    for name, f in kwargs.items():
        if callable(f):
            arr += [f(gb).rename(name)]
        else:
            arr += [reductions[f](gb).rename(name)]

    return pd.concat(arr, axis=1).reset_index()


def groupby_histogram(df, index, column, bins, cumulative=False):
    """Substitute for df.groupby(index)[column].histogram(bins),
    only supports one column label.
    """
    maybe_cumsum = lambda x: x.cumsum(axis=1) if cumulative else x
    column_bin = column + '_bin'
    column_count = column + ('_csum' if cumulative else '_count')
    return (df
        .assign(**{column_bin: bins[np.digitize(df[column], bins) - 1]})
        .pivot_table(index=index, columns=column_bin, values=df.columns[0], 
                     aggfunc='count')
        .reindex(labels=list(bins), axis=1)
        .fillna(0)
        .pipe(maybe_cumsum)
        .stack().rename(column_count)
        .astype(int).reset_index()
           )


def ndarray_to_dataframe(values, index):
    names, levels  = zip(*index)
    columns = pd.MultiIndex.from_product(levels, names=names)
    df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)
    return df


def apply_string_format(df, format_string):
    """Fills in a python string template from column names. Columns
    are automatically cast to string using `.astype(str)`.
    """
    keys = [x[1] for x in Formatter().parse(format_string)]
    result = []
    for values in df[keys].astype(str).values:
        d = dict(zip(keys, values))
        result.append(format_string.format(**d))
    return result


def uncategorize(df, as_codes=False):
    """Pivot and concat are weird with categories.
    """
    for col in df.select_dtypes(include=['category']).columns:
        if as_codes:
            df[col] = df[col].cat.codes
        else:
            df[col] = np.asarray(df[col])
    return df


def natsort_values(df, columns):
    from natsort import natsorted
    data = df[columns].copy()
    uncategorize(data, as_codes=True)
    keys = [(val, i) for i, val in enumerate(data.values)]
    index = [i for _, i in natsorted(keys)]
    return df.iloc[index]


def rank_by_order(df, groupby_columns):
    """Uses 1-based ranking, like `df.groupby(..).rank()`.
    """
    return (df
        .groupby(groupby_columns)
        .apply(lambda x: x.reset_index())
        .reset_index(level=1)['level_1'].pipe(lambda x: list(1 + x)))


def flatten_cols(df, f='_'.join):
    """Flatten column multi index.
    """
    df = df.copy()
    df.columns = [f(x) for x in df.columns]
    return df


def vpipe(df, f, *args, **kwargs):
    """Pipe through a function that accepts and returns a 2D array.

    `df.pipe(vpipe, sklearn.preprocessing.scale)`
    """
    return pd.DataFrame(f(df.values, *args, **kwargs), 
                 columns=df.columns, index=df.index)


# NUMPY
def pile(arr):
    """Concatenate stacks of same dimensionality along leading dimension. Values are
    filled from top left of matrix. Fills background with zero.
    """
    shape = [max(s) for s in zip(*[x.shape for x in arr])]
    # strange numpy limitations
    arr_out = []
    for x in arr:
        y = np.zeros(shape, x.dtype)
        slicer = tuple(slice(None, s) for s in x.shape)
        y[slicer] = x
        arr_out += [y[None, ...]]

    return np.concatenate(arr_out, axis=0)


def montage(arr, shape=None):
    """tile ND arrays ([..., height, width]) in last two dimensions
    first N-2 dimensions must match, tiles are expanded to max height and width
    pads with zero, no spacing
    if shape=(rows, columns) not provided, defaults to square, clipping last row if empty
    """
    sz = list(zip(*[img.shape for img in arr]))
    h, w, n = max(sz[-2]), max(sz[-1]), len(arr)
    if not shape:
        nr = nc = int(np.ceil(np.sqrt(n)))
        if (nr - 1) * nc >= n:
            nr -= 1
    else:
        nr, nc = shape
    M = np.zeros(arr[0].shape[:-2] + (nr * h, nc * w), dtype=arr[0].dtype)

    for (r, c), img in zip(product(range(nr), range(nc)), arr):
        s = [[None] for _ in img.shape]
        s[-2] = (r * h, r * h + img.shape[-2])
        s[-1] = (c * w, c * w + img.shape[-1])
        M[tuple(slice(*x) for x in s)] = img

    return M


def make_tiles(arr, m, n, pad=None):
    """Divide a stack of images into tiles of size m x n. If m or n is between 
    0 and 1, it specifies a fraction of the input size. If pad is specified, the
    value is used to fill in edges, otherwise the tiles may not be equally sized.
    Tiles are returned in a list.
    """
    assert arr.ndim > 1
    h, w = arr.shape[-2:]
    # convert to number of tiles
    m_ = h / m if m >= 1 else int(np.round(1 / m))
    n_ = w / n if n >= 1 else int(np.round(1 / n))

    if pad is not None:
        pad_width = (arr.ndim - 2) * ((0, 0),) + ((0, -h % m), (0, -w % n))
        arr = np.pad(arr, pad_width, 'constant', constant_values=pad)

    h_ = int(int(h / m) * m)
    w_ = int(int(w / n) * n)

    tiled = []
    for x in np.array_split(arr[:h_, :w_], m_, axis=-2):
        for y in np.array_split(x, n_, axis=-1):
            result.append(y)
    
    return tiled


def trim(arr, return_slice=False):
    """Remove i,j area that overlaps a zero value in any leading
    dimension. Trims stitched and piled images.
    """
    def coords_to_slice(i_0, i_1, j_0, j_1):
        return slice(i_0, i_1), slice(j_0, j_1)

    leading_dims = tuple(range(arr.ndim)[:-2])
    mask = (arr == 0).any(axis=leading_dims)
    coords = inscribe(mask)
    sl = (Ellipsis,) + coords_to_slice(*coords)
    if return_slice:
        return sl
    return arr[sl]


@decorator
def applyIJ(f, arr, *args, **kwargs):   
    """Apply a function that expects 2D input to the trailing two
    dimensions of an array. The function must output an array whose shape
    depends only on the input shape. 
    """
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


def inscribe(mask):
    """Guess the largest axis-aligned rectangle inside mask. 
    Rectangle must exclude zero values. Assumes zeros are at the 
    edges, there are no holes, etc. Shrinks the rectangle's most 
    egregious edge at each iteration.
    """
    h, w = mask.shape
    i_0, i_1 = 0, h - 1
    j_0, j_1 = 0, w - 1
    
    def edge_costs(i_0, i_1, j_0, j_1):
        a = mask[i_0, j_0:j_1 + 1].sum()
        b = mask[i_1, j_0:j_1 + 1].sum()
        c = mask[i_0:i_1 + 1, j_0].sum()
        d = mask[i_0:i_1 + 1, j_1].sum()  
        return a,b,c,d
    
    def area(i_0, i_1, j_0, j_1):
        return (i_1 - i_0) * (j_1 - j_0)
    
    coords = [i_0, i_1, j_0, j_1]
    while area(*coords) > 0:
        costs = edge_costs(*coords)
        if sum(costs) == 0:
            return coords
        worst = costs.index(max(costs))
        coords[worst] += 1 if worst in (0, 2) else -1
    return


def subimage(stack, bbox, pad=0):
    """Index rectangular region from [...xYxX] stack with optional constant-width padding.
    Boundary is supplied as (min_row, min_col, max_row, max_col).
    If boundary lies outside stack, raises error.
    If padded rectangle extends outside stack, fills with fill_value.

    bbox can be bbox or iterable of bbox (faster if padding)
    :return:
    """ 
    i0, j0, i1, j1 = bbox + np.array([-pad, -pad, pad, pad])

    sub = np.zeros(stack.shape[:-2]+(i1-i0, j1-j0), dtype=stack.dtype)

    i0_, j0_ = max(i0, 0), max(j0, 0)
    i1_, j1_ = min(i1, stack.shape[-2]), min(j1, stack.shape[-1])
    s = (Ellipsis, 
         slice(i0_-i0, (i0_-i0) + i1_-i0_),
         slice(j0_-j0, (j0_-j0) + j1_-j0_))

    sub[s] = stack[..., i0_:i1_, j0_:j1_]
    return sub


def offset(stack, offsets):
    """Applies offset to stack, fills with zero. Only applies integer offsets.
    """
    if len(offsets) != stack.ndim:
        if len(offsets) == 2 and stack.ndim > 2:
            offsets = [0] * (stack.ndim - 2) + list(offsets)
        else:
            raise IndexError("number of offsets must equal stack dimensions, or 2 (trailing dimensions)")

    offsets = np.array(offsets).astype(int)

    n = stack.ndim
    ns = (slice(None),)
    for d, offset in enumerate(offsets):
        stack = np.roll(stack, offset, axis=d)
        if offset < 0:
            index = ns * d + (slice(offset, None),) + ns * (n - d - 1)
            stack[index] = 0
        if offset > 0:
            index = ns * d + (slice(None, offset),) + ns * (n - d - 1)
            stack[index] = 0

    return stack    


def join_stacks(*args):
    def with_default(arg):
        try:
            arr, code = arg
            return arr, code
        except ValueError:
            return arg, ''

    def expand_dims(arr, n):
        if arr.ndim < n:
            return expand_dims(arr[None], n)
        return arr

    def expand_code(arr, code):
        return code + '.' * (arr.ndim - len(code))

    def validate_code(arr, code):
        if code.count('a') > 1:
            raise ValueError('cannot append same array along multiple dimensions')
        if len(code) > arr.ndim:
            raise ValueError('length of code greater than number of dimensions')

    def mark_all_appends(codes):
        arr = []
        for pos in zip(*codes):
            if 'a' in pos:
                if 'r' in pos:
                    raise ValueError('cannot repeat and append along the same axis')
                pos = 'a' * len(pos)
            arr += [pos]
        return [''.join(code) for code in zip(*arr)]

    def special_case_no_ops(args):
        if all([c == '.' for _, code in args for c in code]):
            return [(arr[None], 'a' + code) for arr, code in args]
        return args
    
    # insert default code (only dots)
    args = [with_default(arg) for arg in args]
    # expand the dimensions of the input arrays
    output_ndim = max(arr.ndim for arr, _ in args)
    args = [(expand_dims(arr, output_ndim), code) for arr, code in args]
    # add trailing dots to codes
    args = [(arr, expand_code(arr, code)) for arr, code in args]
    # if no codes are provided, interpret as appending along a new dimension
    args = special_case_no_ops(args)
    # recalculate due to special case
    output_ndim = max(arr.ndim for arr, _ in args)
    
    [validate_code(arr, code) for arr, code in args]
    # if any array is appended along an axis, every array must be
    # input codes are converted from dot to append for those axes
    codes = mark_all_appends([code for _, code in args])
    args = [(arr, code) for (arr, _), code in zip(args, codes)]

    # calculate shape for output array
    # uses numpy addition rule to determine output dtype
    output_dtype = sum([arr.flat[:1] for arr, _ in args]).dtype
    output_shape = [0] * output_ndim
    for arr, code in args:
        for i, c in enumerate(code):
            s = arr.shape[i]
            if c == '.':
                if output_shape[i] == 0 or output_shape[i] == s:
                    output_shape[i] = s
                else:
                    error = 'inconsistent shapes {0}, {1} at axis {2}'
                    raise ValueError(error.format(output_shape[i], s, i))

    for arg, code in args:
        for i, c in enumerate(code):
            s = arg.shape[i]
            if c == 'a':
                output_shape[i] += s
    
    output = np.zeros(output_shape, dtype=output_dtype)
    
    # assign from input arrays to output 
    # (values automatically coerced to most general numeric type)
    slices_so_far = [0] * output_ndim
    for arr, code in args:
        slices = []
        for i, c in enumerate(code):
            if c in 'r.':
                slices += [slice(None)]
            if c == 'a':
                s = slices_so_far[i]
                slices += [slice(s, s + arr.shape[i])]
                slices_so_far[i] += arr.shape[i]

        output[tuple(slices)] = arr
        
    return output

# SCIKIT-IMAGE
def regionprops(labeled, intensity_image):
    """Supplement skimage.measure.regionprops with additional field `intensity_image_full` 
    containing multi-dimensional intensity image.
    """
    import skimage.measure

    if intensity_image.ndim == 2:
        base_image = intensity_image
    else:
        base_image = intensity_image[..., 0, :, :]

    regions = skimage.measure.regionprops(labeled, intensity_image=base_image)

    for region in regions:
        b = region.bbox
        region.intensity_image_full = intensity_image[..., b[0]:b[2], b[1]:b[3]]

    return regions