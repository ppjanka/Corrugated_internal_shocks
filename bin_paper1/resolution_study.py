
# coding: utf-8

# In[ ]:


import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import glob
import sys
from tqdm import tqdm
from scipy import signal
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

import socket
import __main__
machine = socket.gethostname()
print('Running on machine %s' % machine)
in_script = hasattr(__main__, '__file__')

if in_script:
    pathstem = '/cfs/klemming/projects/snic/snic2020-4-12/ppjanka/intsh2/bin_paper1'
    nproc = int(sys.argv[1])
    save = True
else: # notebook run
    pathstem = '/DATA/Dropbox/LOOTRPV/astro_projects/2020_IntSh2/athena4p2/bin_paper1'
    nproc = 10
    save = False
    
if nproc < 0:
    import multiprocessing
    nproc = multiprocessing.cpu_count()


# In[ ]:


def measure_shock_single (filename_2d, margin, dvel_threshold, gs=None):
    
    try:

        # load the snapshot
        with open(filename_2d, 'rb') as f:
            data = pkl.load(f)[0]

        # basic properties
        dx = data['x1v'][1] - data['x1v'][0]
        middle = len(data['x1v']) // 2
        idx_margin = int(margin / dx)

        # find the shock front by local extrema of dvx/dx
        vel = data['vel1'][:,(middle-idx_margin):(middle+idx_margin)]
        dvel = signal.convolve2d(vel, [[-1,1]], mode='valid')
        xavg = 0.5*(data['x1v'][(middle-idx_margin+1):(middle+idx_margin)] + data['x1v'][(middle-idx_margin):(middle+idx_margin-1)])

        shock_y, shock_x = list(map(np.array, np.where(np.abs(dvel) > dvel_threshold)))
        df_shocks = pd.DataFrame(data=np.transpose((shock_y, shock_x)), columns=('shock_y', 'shock_x'))

        # limit shock_x to two values -- for the left and right shock
        shock_xl = df_shocks[df_shocks.shock_x < idx_margin].groupby('shock_y').mean()
        shock_xr = df_shocks[df_shocks.shock_x > idx_margin].groupby('shock_y').mean()
        shock_xlr = shock_xl.join(
            shock_xr,
            how='inner',
            lsuffix='l',
            rsuffix='r'
        )
        shock_y = shock_xlr.index

        # calculate the shock centroid
        shock_x = 0.5 * (shock_xlr.shock_xl + shock_xlr.shock_xr)
        
        # clean up
        del df_shocks, shock_xl, shock_xr

        if gs:
            plt.subplot(gs[2,0])
            plt.plot(xavg, dvel[10,:])
            plt.xlabel('x [sim.u.]')
            plt.ylabel('dv/dx [sim.u.]')
            plt.xlim(-margin,margin)

            plt.subplot(gs[2,1])
            plt.contourf(data['x1v'], data['x2v'], data['rho'], 64)
            plt.title('2D density')
            plt.scatter(
                [xavg[x] for x in shock_xlr.shock_xl.astype(int)],
                [data['x2v'][y] for y in shock_y.astype(int)],
                color='r', s=0.05
            )
            plt.scatter(
                [xavg[x] for x in shock_xlr.shock_xr.astype(int)],
                [data['x2v'][y] for y in shock_y.astype(int)],
                color='r', s=0.05
            )

            plt.scatter(
                [xavg[x] for x in shock_x.astype(int)],
                [data['x2v'][y] for y in shock_y.astype(int)],
                color='g', s=0.05
            )
            plt.xlim(-1.25,1.25)

            # calculate corrugation resolution
            plt.subplot(gs[3,:])
        nbin = max(10, int(np.max(shock_x)-np.min(shock_x)/dx))
        n, bins, patches = plt.hist(shock_x, )

        bins = 0.5*(bins[1:] + bins[:-1])
        shock_x = bins[np.where(n > np.mean(n))[0]]
        shock_edges = np.min(shock_x), np.max(shock_x)

        for e in shock_edges:
            plt.axvline(e, color='r')

        shock_width = shock_edges[1] - shock_edges[0]

        if gs:
            print(f' - shock width is {shock_width:.1f}', flush=True)
        
    except FileNotFoundError:
        shock_width = -1.0
        print(' - 2D snapshot not found. Continuing.', flush=True)
    
    return shock_width

def measure_shocks (suite,
                    version='prod1_corr_ampl',
                    amplitudes=(1,2,5,10,20,50,75,100),
                    all_2d=False,
                    margin=2.0, # how far from the center do we expect the shock
                    dvel_threshold=0.1,
                    shell_width=1.0, dt=0.1,
                    time_factor=(1.0,2.0),
                    save=False
    ):
    
    amplitudes_here = []
    shock_widths = []
    
    for ampl in amplitudes:

        print(f'Amplitude: {ampl}', flush=True)

        plt.figure(figsize=(8,12))
        gs = GridSpec(4,2)

        # first, find the collision time using 1D data

        rhomax = []
        rho_prev = np.inf; i_start = 0
        for i in tqdm(range(len(os.listdir(f'{pathstem}/{suite}/{version}/results_corr0ampl{ampl}/joined_vtk')))):
            with open(f'{pathstem}/{suite}/prod1_corr_ampl/results_corr0ampl{ampl}/joined_vtk/IntSh2-p1.{i:04d}.vtk.pkl', 'rb') as f:
                data = pkl.load(f)[0]
            rho = np.max(data['rho'])
            if rho > rho_prev: # ignore the initial decompression
                rhomax.append(rho)
                rho_prev = -np.inf
                if not i_start: i_start = i
            else:
                rho_prev = rho
                continue
            del data

        i_coll = np.argmax(rhomax)+i_start
        print(f' - maximum shock compression at i = {i_coll}', flush=True)
        di = (np.array(time_factor) * shell_width / dt).astype(int)
        print(f' - corrugation should be measured for i within {i_coll - di[0]} -- {i_coll + di[1]}')

        plt.subplot(gs[0,0])
        plt.plot(range(len(rhomax)), rhomax)
        plt.xlabel('Frame no.')
        plt.ylabel('Maximum density [sim.u.]')
        plt.axvline(i_coll-i_start, color='r')
        plt.text(i_coll-i_start, min(rhomax), i_coll, color='r')

        with open(f'{pathstem}/{suite}/{version}/results_corr0ampl{ampl}/joined_vtk/IntSh2-p1.{i_coll:04d}.vtk.pkl', 'rb') as f:
            data = pkl.load(f)[0]
        rho = np.max(data['rho'])

        plt.subplot(gs[0,1])
        plt.contourf(data['x1v'], data['x2v'], data['rho'], 64)
        plt.title('1D density')
        plt.xlim(-1,1)

        # then, fine-tune on 2D data ------------------------------------------
        if all_2d:
            filenames_2d = [f'{pathstem}/{suite}/{version}/results_corr1ampl{ampl}/joined_vtk/IntSh2-p1.{i:04d}.vtk.pkl' for i in range(i_coll-di[0], i_coll+di[1]+1)]
        else:
            filenames_2d = [f'{pathstem}/{suite}/{version}/results_corr1ampl{ampl}/joined_vtk/IntSh2-p1.{i_coll:04d}.vtk.pkl']
        
        rhomax = []
        rho_prev = np.inf; i_start = 0

        if nproc > 1:
            with Pool(nproc) as pool:
                shock_widths_here = pool.map(
                    measure_shock_single,
                    filenames_2d,
                    ([margin,]*len(filenames_2d)),
                    ([dvel_threshold,]*len(filenames_2d))
                )
        else:
            shock_widths_here = []
            for filename_2d in tqdm(filenames_2d):
                shock_widths.append(
                    measure_shock_single(
                        filename_2d,
                        margin=margin,
                        dvel_threshold=dvel_threshold
                    )
                )
        frames_here = np.arange(len(shock_widths_here))
            
        if len(frames_here) > 0:
            plt.subplot(gs[1,:])
            plt.plot(frames_here, shock_widths_here)
            plt.ylabel('Shock width [px]')
            plt.xlabel('2D snapshot no.')
            plt.axhline(0, color='k', linewidth=0.5)

            shock_width_idx = np.argmax(shock_widths_here)
            shock_width = measure_shock_single(
                    filenames_2d[shock_width_idx],
                    margin=margin,
                    dvel_threshold=dvel_threshold,
                    gs=gs
                )

            shock_widths.append(shock_width)
            amplitudes_here.append(ampl)
            
        # add frame information
        plt.suptitle(f'Amplitude: {ampl}, shock width: {shock_width:.1f}')
            
        # show the plot
        plt.tight_layout()
        if save:
            plt.savefig(f'{pathstem}/{suite}/{version}/shock_width_ampl{ampl}.png', format='png', dpi=300)
            with open(f'{pathstem}/{suite}/{version}/shock_width_ampl{ampl}.pkl', 'wb') as f:
                pkl.dump((frames_here, shock_widths_here), f)
        else:
            plt.show()
        plt.close()
        
    plt.plot(amplitudes_here, shock_widths)
    plt.xlabel('Amplitude')
    plt.ylabel('Shock width [px]')
    if save:
        plt.savefig(f'{pathstem}/{suite}/{version}/shock_widths.png', format='png', dpi=300)
        with open(f'{pathstem}/{suite}/{version}/shock_widths.pkl', 'wb') as f:
            pkl.dump((amplitudes_here, shock_widths), f)
    else:
        plt.show()
    plt.close()

    print('Done.', flush=True)


# In[ ]:


measure_shocks('corrT1_dens', all_2d=True, save=save)


# In[ ]:


measure_shocks('corrT2_press', all_2d=True, time_factor=(1,4), margin=4.0, save=save)


# In[ ]:


if not save:
    with open('/DATA/Dropbox/LOOTRPV/astro_projects/2020_IntSh2/athena4p2/bin_paper1/corrT2_press/prod1_corr_ampl/results_corr1ampl20/joined_vtk/IntSh2-p1.0037.vtk.pkl', 'rb') as f:
        data, _ = pkl.load(f)

    plt.figure(figsize=(12,12))
    plt.contourf(data['x1v'], data['x2v'], data['rho'], bins=64)
    plt.xlim(-0.1,0.1)
    plt.show(); plt.close()


# In[ ]:




