# This file contains functions to calculate and plot cartographic projections of data defined on a sphere.
# Author: Patryk Pjanka, 2021

import numpy as np
from tqdm import tqdm

def projection_EckertIV (theta, phi, tolerance=1.0e-3):
    # calculate the theta_eckert parameter
    from scipy.optimize import root
    theta_eckert = root(lambda x : x+np.sin(x)*np.cos(x)+2*np.sin(x)-(2+0.5*np.pi)*np.sin(0.5*np.pi-theta), x0=np.zeros(theta.shape), tol=tolerance)
    theta_eckert = np.where(theta_eckert.fun < tolerance, theta_eckert.x, np.nan*np.ones(theta_eckert.x.shape))
    # calculate the projected on-image coordinates (R=1 assumed)
    x = 2./np.sqrt(4.*np.pi+np.pi*np.pi)*phi*(1.+np.cos(theta_eckert))
    y = 2.*np.sqrt(np.pi/(4.+np.pi))*np.sin(theta_eckert)
    return x,y

def plot_cartographic_grid (ax, projection, theta_ticks=[], phi_ticks=[], in_unit='deg', tick_unit='deg', res=50, style_dict={'color':'k','ls':'-','lw':1}):
    # process the input, prepare for calculations
    if in_unit == 'deg':
        _phi_ticks = np.array(phi_ticks) * np.pi / 180.
        _theta_ticks = np.array(theta_ticks) * np.pi / 180.
    elif in_unit == 'rad':
        _phi_ticks = np.array(phi_ticks)
        _theta_ticks = np.array(theta_ticks)
    else:
        print('[plot_cartographic_grid]: Input unit not recognized! Aborted.')
        return
    _phi_ticks = list(_phi_ticks) + [-np.pi, np.pi]
    _theta_ticks = list(_theta_ticks) + [0., np.pi]
    argspace = np.linspace(0.,1.,res)
    # plot the grid
    for phi in _phi_ticks:
        _theta = np.pi*argspace
        _phi = phi*np.ones(res)
        x, y = projection(_theta, _phi)
        ax.plot(x,y,**style_dict)
    for theta in _theta_ticks:
        _theta = theta * np.ones(res)
        _phi = -np.pi + 2.*np.pi*argspace
        x, y = projection(_theta, _phi)
        ax.plot(x,y,**style_dict)
    # remove the default plt annotations
    ax.set_axis_off()
    
def add_patch (ax, projection, theta_range=[0.,180.], phi_range=[-180.,180.], in_unit='deg', color='r', res=8):
    # process the input, prepare for calculations
    if in_unit == 'deg':
        _phi_range = np.array(phi_range) * np.pi / 180.
        _theta_range = np.array(theta_range) * np.pi / 180.
    elif in_unit == 'rad':
        _phi_range = np.array(phi_range)
        _theta_range = np.array(theta_range)
    else:
        print('[add_patch]: Input unit not recognized! Aborted.')
        return
    # generate the patch outline in projection
    phi_diff = _phi_range[1] - _phi_range[0]
    theta_diff = _theta_range[1] - _theta_range[0]
    argspace = np.linspace(0.,1.,res)
    x,y = projection(_theta_range[0]*np.ones(res), _phi_range[0]+phi_diff*argspace)
    outline = np.array([x,y]).transpose()
    x,y = projection(_theta_range[0]+theta_diff*argspace, _phi_range[1]*np.ones(res))
    outline = np.concatenate([outline, np.array([x,y]).transpose()])
    x,y = projection(_theta_range[1]*np.ones(res), _phi_range[1]-phi_diff*argspace)
    outline = np.concatenate([outline, np.array([x,y]).transpose()])
    x,y = projection(_theta_range[1]-theta_diff*argspace, _phi_range[0]*np.ones(res))
    outline = np.concatenate([outline, np.array([x,y]).transpose()])
    # add the patch to the plot
    from matplotlib.patches import Polygon
    poly = Polygon(outline, color=color)
    ax.add_artist(poly)

def plot_projected (ax, projection, thetas, phis, bin_vals, in_unit='rad', cmap='rainbow', res_patch=8):

    from matplotlib import cm
    _cmap = cm.get_cmap(cmap)

    # normalize the color scale
    vmin = np.nanmin(bin_vals)
    vmax = np.nanmax(bin_vals)
    colors = (bin_vals - vmin) / (vmax-vmin)

    for j in tqdm(range(bin_vals.shape[0])):
        for k in range(bin_vals.shape[1]):
            add_patch(ax, projection, phi_range=phis[k:(k+2)], theta_range=thetas[j:(j+2)], in_unit=in_unit, color=_cmap(colors[j,k]), res=res_patch)