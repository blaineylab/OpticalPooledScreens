import re
import os
from ops.constants import FILE


FILE_PATTERN = [
        r'((?P<home>.*)\/)?',
        r'(?P<dataset>(?P<date>[0-9]{8}).*?)\/',
        r'(?:(?P<subdir>.*)\/)*',
        r'(MAX_)?(?P<mag>[0-9]+X).',
        r'(?:(?P<cycle>[^_\.]*).*?(?:.*MMStack)?.)?',
        r'(?P<well>[A-H][0-9]*)',
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',
        r'(?:_Tile-(?P<tile>([0-9]+)))?',
        r'(?:\.(?P<tag>.*))*\.(?P<ext>tif|pkl|csv|fastq)']

folder_pattern = [
        r'(?P<mag>[0-9]+X).',
        r'(?:(?P<cycle>[^_\.]*).*?)\/',
        r'(?P<well>[A-H][0-9]+)',
        r'(?:[_-]Site[_-](?P<site>([0-9]+)))?',
        r'\/?']

FILE_PATTERN_ABS = ''.join(FILE_PATTERN)
FILE_PATTERN_REL = ''.join(FILE_PATTERN[2:])
        
FOLDER_PATTERN_ABS = ''.join(FILE_PATTERN[:2] + folder_pattern)
folder_pattern_rel = ''.join(folder_pattern)


def parse_filename(filename):
    """Parse filename into dictionary. 

    Some entries in the dictionary are optional, e.g., cycle and tile.

    Examples:
        >>> parse('example_data/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-107.max.tif')

        {'subdir': 'example_data/input/10X_c1-SBS-1',
         'mag': '10X',
         'cycle': 'c1-SBS-1',
         'well': 'A1',
         'tile': '107',
         'tag': 'max',
         'ext': 'tif',
         'file': 'example_data/input/10X_c1-SBS-1/10X_c1-SBS-1_A1_Tile-107.max.tif'}
    """
    filename = os.path.normpath(filename)
    filename = filename.replace('\\', '/')

    patterns = FILE_PATTERN_ABS, FILE_PATTERN_REL, FOLDER_PATTERN_ABS, folder_pattern_rel

    for pattern in patterns:
        match = re.match(pattern, filename)
        try:
            result = {k:v for k,v in match.groupdict().items() if v is not None}
            result[FILE] = filename  # convenience, not used by name_file
            return result
        except AttributeError:
            continue
    
    raise ValueError('failed to parse filename: %s' % filename)


def name_file(description, **more_description):
    """Name a file from a dictionary of filename parts. 

    Can override dictionary with keyword arguments.
    """
    d = dict(description)

    for k, v in more_description.items():
        if v is None and k in d:
            d.pop(k)
        else:
            d[k] = v

    d = {k: v for k,v in d.items() if v is not None}

    assert 'tag' in d

    if 'cycle' in d:
        a = '%s_%s_%s' % (d['mag'], d['cycle'], d['well'])
    else:
        a = '%s_%s' % (d['mag'], d['well'])

    # only one
    if 'tile' in d:
        b = 'Tile-%s' % d['tile']
    elif 'site' in d:
        b = 'Site-%s' % d['site']
    else:
        b = None

    if b:
        basename = '%s_%s.%s.%s' % (a, b, d['tag'], d['ext'])
    else:
        basename = '%s.%s.%s' % (a, d['tag'], d['ext'])
    
    optional = lambda x: d.get(x, '')
    filename = os.path.join(optional('home'), optional('dataset'), optional('subdir'), basename)
    return os.path.normpath(filename)

def guess_filename(row, tag, **override_fields):
    description = {'subdir': 'process', 'mag': '10X', 
                    'tag': tag, 'ext': 'tif'}
    description.update(row.to_dict())
    description.update(override_fields)
    return name_file(description)

