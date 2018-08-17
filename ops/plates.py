import string
import numpy as np


def add_global_xy(df, well_spacing, grid_shape, grid_spacing='10X', factor=1.):
    """Adds global x and y coordinates to a dataframe with 
    columns indicating (i, j) or (x, y) positions. 
    """
    
    df = df.copy()
    wt = list(zip(df['well'], df['tile']))
    d = {(w,t): plate_coordinate(w, t, well_spacing, grid_spacing, grid_shape) for w,t in set(wt)}
    y, x = zip(*[d[k] for k in wt])

    if 'x' in df:
        df['global_x'] = x + df['x'] * factor
        df['global_y'] = y + df['y'] * factor
    elif 'i' in df:
        df['global_x'] = x + df['j'] * factor
        df['global_y'] = y + df['i'] * factor
    else:
        df['global_x'] = x
        df['global_y'] = y

    df['global_y'] *= -1
    return df


def plate_coordinate(well, tile, well_spacing, grid_spacing, grid_shape):
    """Returns global plate coordinate (i,j) in microns for a tile in a well. 
    The position is based on:
    `well_spacing` microns
      or one of '96w', '24w', '6w' for standard well plates 
    `grid_spacing` microns
      or one of '10X', '20X' for common spacing at a given magnification
    `grid_shape` (# rows, # columns)  
    """
    tile = int(tile)
    if well_spacing == '96w':
        well_spacing = 9000
    if well_spacing == '24w':
        well_spacing = 19300
    if well_spacing == '6w':
        well_spacing = 39120
        
    if grid_spacing == '10X':
        delta = 1280
    elif grid_spacing == '20X':
        delta = 640
    else:
        delta = grid_spacing

    row, col = well_to_row_col(well)
    i, j = row * well_spacing, col * well_spacing

    height, width = grid_shape
    i += delta * int(tile / width)
    j += delta * (tile % width)
    
    i -= delta * ((height - 1) / 2.) 
    j -= delta * ((width  - 1)  / 2.)

    return i, j


def well_to_row_col(well):
    return string.ascii_uppercase.index(well[0]), int(well[1:]) - 1


def remap_snake(site, grid_shape):
    """Maps site names from snake order (Micro-Manager HCS plugin) 
    to row order.
    """

    rows, cols = grid_shape
    grid = np.arange(rows*cols).reshape(rows, cols)
    grid[1::2] = grid[1::2, ::-1]
    site_ = grid.flat[int(site)]
    return '%d' % site_

