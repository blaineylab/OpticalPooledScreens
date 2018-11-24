import pandas as pd
import numpy as np

def load_hist(filename, threshold=3):
    try:
        return (pd.read_csv(filename, sep='\s+', header=None)
            .rename(columns={0: 'count', 1: 'seq'})
            .query('count > @threshold')
            .assign(fraction=lambda x: x['count']/x['count'].sum())
            .assign(log10_fraction=lambda x: np.log10(x['fraction']))
            .assign(file=filename)
           )
    except pd.errors.EmptyDataError:
        return None


def calc_stats(df_hist, df_design):
    sizes = df_design.drop_duplicates('oligo').groupby('group').size()

    fraction_stat = (df_design
     .drop_duplicates('oligo')
     .merge(df_hist, how='left')
     .groupby(['group', 'dialout', 'well'])
         ['fraction'].sum().reset_index()
    )


    main_stat = (df_design
     .drop_duplicates('oligo')
     .merge(df_hist, how='left')
     .groupby(['group', 'dialout', 'well'])['count']
     .describe(percentiles=[0.1, 0.5, 0.9])
     .rename(columns={'10%': 'Q10', 
                      '50%': 'Q50', 
                      '90%': 'Q90'})
     .join(sizes.rename('designed'), on='group')
     .assign(ratio_90_10=lambda x: x.eval('Q90 / Q10'))
     .assign(missing=lambda x: x.eval('designed - count').astype(int))
     .sort_index(level='well')
     
    )     
    return (main_stat.reset_index()
        .merge(fraction_stat)
        .set_index(['group', 'dialout', 'well']))