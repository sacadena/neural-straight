'''
Utility visualization functions for neural straightening package
Author: Santiago Cadena
email: santiago.cadena@uni-tuebingen.de
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

def vis_pca_trajectory(x):
    '''
    Plots the trajectory of frames projected into the first two principal components
    Args:
        :x: a array with where the first axis represents t_steps
    Returns:
        :fig:  a matplotlib figure with the trajectory
        :Xred: an array with the data projected into the first two components
    '''
    T = x.shape[0]
    X = x.reshape(T, -1)
    
    # Do PCA and project signals
    pca = PCA(n_components=2)  
    pca.fit(X)                
    Xred = pca.transform(X)
    
    # Create figure object
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    sns.set_context('paper', font_scale=1.1)
    with sns.axes_style("ticks"):
        ax.plot(Xred[:,0], Xred[:,1], 'k-')
        ax.scatter(Xred[:,0], Xred[:,1], c = np.arange(T), cmap=plt.cm.Reds, alpha =1)
    sns.despine(trim =True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.set_dpi(100)
    return fig, Xred

def view_frames(x):
    '''
    Visualizes some frames of a movie
    Args:
        :x: array with t_Steps x width x height
    '''
    sns.set_context('paper', font_scale=1.1)
    fig, axes = plt.subplots(2,5,figsize = (20,5))
    with sns.axes_style("ticks"):
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(x[i*10, ], cmap = 'gray')
            ax.axis('off')

            
#-------- For response scatter visualizing 

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


def make_df_areas(df, metric='average', order_areas = None):
    '''
    Creates a dataframe with the metric values for each bran area a s columns
    Args:
        :df: Data frame created from the datajoint relation containing avg_curvature and median_curvature and brain_area
        :metric: string. Either "average" or "median"
        :order_areas: a list with the brain areas
    Returns:
        :df_area: The dataframe with brain areas useful for Pariwise scatterplot
    '''
    
    assert metric in ['average', 'median'], 'Metric should be average or median'
    
    if order_areas is None:
        order_areas = ['V1', 'LM', 'LI', 'AL', 'LLA','P', 'POR','RL']
    
    key_metric = ['avg_curvature' if metric == 'average' else 'median_curvature'][0]
    
    # Make new dataframes of metrics and areas
    df_area = dict()
    for a in order_areas:
        curvature   = df[df.brain_area == a][key_metric].values
        df_area[a]  = curvature
    df_area = pd.DataFrame(df_area)
    
    return df_area
    
    

def scatter_brain_areas(df, metric='average', order_areas=None, **kwargs):
    '''
    Plots scatter plot pf brain area responses
    Args:
        :df: Data frame created from the datajoint relation containing avg_curvature and median_curvature and brain_area
        :metric: string. Either "average" or "median"
        :order_areas: a list with the brain areas
    '''
    
    assert metric in ['average', 'median'], 'Metric should be average or median'
    #key_metric = ['avg_curvature' if metric == 'average' else 'median_curvature'][0]
    
    df_area = make_df_areas(df, metric='average', order_areas = order_areas)
    
    # Scatter pair plot
    sns.set_context('paper', font_scale=1.1)
    with sns.axes_style("ticks"):
        g = sns.PairGrid(data=df_area, **kwargs)
        g.map_lower(sns.scatterplot, color = 'k', s=2, linewidth=0.1)
        
    vmin = df_area.values.min()
    vmax = df_area.values.max()
    
    for ax in g.axes.flatten():
#         if metric=='median':
#             ax.plot([18, 23], [18, 23], '--', color = 'gray')
#         else:
#             ax.plot([18, 25], [18, 25], '--', color = 'gray')
            
        ax.plot([vmin, vmax], [vmin, vmax], '--', color = 'gray')
    
    g.map_upper(hide_current_axis)  # hide upper diagonal
    g.fig.set_size_inches(8, 8)
    plt.suptitle('{} Curvature (°)'.format(metric.capitalize()))
    return g.fig


def brain_area_curvature(df, metric='average', order_areas=None, **kwargs):
    '''
    Plots the summary performance for each brain area
    Args:
        :df: Data frame created from the datajoint relation containing avg_curvature and median_curvature and brain_area
        :metric: string. Either "average" or "median"
        :order_areas: a list with the brain areas
    '''
    
    assert metric in ['average', 'median'], 'Metric should be average or median'
    key_metric = ['avg_curvature' if metric == 'average' else 'median_curvature'][0]
    if order_areas is None:
        order_areas = ['V1', 'LM', 'LI', 'AL', 'LLA','P', 'POR','RL']
    
    sns.set_context('paper', font_scale=1.1)
    sns.set_palette(sns.xkcd_palette(['grey', 'golden yellow']))
    with sns.axes_style("ticks"):
        g = sns.catplot('brain_area', key_metric, data = df, order = order_areas, **kwargs)
    sns.despine(trim=True)
    g.ax.set_xlabel('Brain area')
    g.ax.set_ylabel('{} Curvature (°)'.format(metric.capitalize()))
    g.fig.set_size_inches(4,4)
    return g.fig

            
#-------- For curvature instensity space visualization


def histogram_object_types(df, metric='average', order_types=None, **kwargs):
    '''
    Plots a histogram for the curvature metric for each movie type in one plot. adds plots for each filter type
    Args:
        :df: Data frame created from the datajoint relation containing avg_curvature and median_curvature, filter_type, movie_type
        :metric: string. Either "average" or "median"
        :order_types: a list with the types of movies
    '''
    assert metric in ['average', 'median'], 'Metric should be average or median'
    key_metric = ['avg_pixel_curvature' if metric == 'average' else 'median_pixel_curvature'][0]
    
    if order_types is None: order_types = ['type1', 'type2', 'type3']
    
    sns.set_context('paper', font_scale=1.1)
    sns.set_palette(sns.xkcd_palette(['grey', 'golden yellow']))
    
    with sns.axes_style("ticks"):
        g = sns.FacetGrid(df, hue = 'type_movie', col = 'filter_type', legend_out=True, hue_order = order_types)
        g = (g.map(sns.distplot, key_metric, kde=False, **kwargs)).add_legend(title='Movie type')
        g.set_titles("{col_name} filter")
        g.set_xlabels('{} pixel curvature (°)'.format(metric.capitalize()))

        g.axes[0,0].set_ylabel('Counts')
        sns.despine(trim=False)
        g.fig.set_dpi(100)
        g.fig.set_size_inches(5,3)
    
    return g.fig


# ------------------ scatter of pixel vs response curvature:

def scatter_pix_responses(df_px_resp, metric='average', order_areas=None, order_types=None, **kwargs):
    '''
    Plots a scatter plot of curvature in pixel space vs response space and color codes the clips based on type
    Args:
        :df: Data frame created from the datajoint relation containing avg_curvature and median_curvature, filter_type, movie_type
        :metric: string. Either "average" or "median"
        :order_areas: a list with the brain areas
        :order_types: a list with the types of movies
    '''
    assert metric in ['average', 'median'], 'Metric should be average or median'
    key_metric_px = ['avg_pixel_curvature' if metric == 'average' else 'median_pixel_curvature'][0]
    key_metric    = ['avg_curvature' if metric == 'average' else 'median_curvature'][0]
    
    
    if order_types is None: order_types = ['type1', 'type2', 'type3']
    if order_areas is None: order_areas = ['V1', 'LM', 'LI', 'AL', 'LLA','P', 'POR','RL']
        
    sns.set_context('paper', font_scale=1.1)
    sns.set_palette(sns.xkcd_palette(['grey', 'golden yellow']))
    with sns.axes_style("ticks"):
        g = sns.FacetGrid(df_px_resp, col='brain_area', hue = 'type_movie', aspect=1, col_order = order_areas, hue_order = order_types, col_wrap = 4, \
                         legend_out=True, margin_titles=True)
        g = (g.map(sns.scatterplot, key_metric_px, key_metric, alpha = 1, s = 3, linewidth = 0.1).add_legend(title = 'Movie Type'))
        g.set_titles("{col_name}")
        g.set_xlabels('')
        g.set_ylabels('{}. Response Curvature (°)'.format(metric.capitalize()[:3]))
    for ax in g.axes.flatten():
        ax.plot([15, 28], [15, 28], '--k')
        ax.set_aspect('equal')
    ax.set_xlabel('{}. Pixel Curvature (°)'.format(metric.capitalize()[:3]))
    sns.despine(trim=True)
    g.fig.set_size_inches(8, 5)
    
    return g.fig