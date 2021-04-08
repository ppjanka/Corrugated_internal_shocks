# This is a script with instructions to read, process, and plot the .lis outputs from the Athena 4.2 particle module.
# The structure is adapted (by P. Pjanka, 2021) from the *.m files within this folder.

from pathos.pools import ProcessPool # an alternative to Python's multiprocessing

import numpy as np
from scipy.signal import convolve as sp_convolve

import struct # for reading binary data
import sys
import os
import glob
from tqdm import tqdm # progress bar

class Particles:
    
    def __init__ (self, relativistic=False): # empty class
        self.initialized = False
        self.relativistic = relativistic
        self.npart = 0
        self.nparttypes = 0
        self.coords = {}
        self.times = []
        self.dts = []
        # arrays with particle data:
        #   [ dimensions: <time> <particle> <quantity> ]
        self.pos = []
        self.vel = []
        self.dpar = []
        self.grp = []
        self.my_id = []
        self.init_id = []
        # auxiliary data calculated from the above
        self.Ekin = []
        
    def process_metadata (self):
        # Read the coordinate limits
        x1l, x1u, x2l, x2u, x3l, x3u, \
        x1dl, x1du, x2dl, x2du, x3dl, x3du = struct.unpack('f'*12, self.file.read(4*12))
        coords = {'x1min': x1l, 'x1max': x1u, \
                  'x2min': x2l, 'x2max': x2u, \
                  'x3min': x3l, 'x3max': x3u}
        # Read particle property info
        nparttypes, = struct.unpack('i', self.file.read(4))
        grproperty = []
        for pt in range(nparttypes):
            grproperty.append(struct.unpack('f', self.file.read(4))[0])
        # Read time
        time, dt = struct.unpack('f'*2, self.file.read(4*2))
        # Read particle number
        npart, = struct.unpack('l', self.file.read(8))
        if self.initialized: # check for consistency
            if self.npart != npart or self.nparttypes != self.nparttypes or self.coords != coords:
                print('ERROR: metadata mismatch, aborting.', flush=True)
                sys.exit(1)
        else: # initialize metadata
            self.npart = npart
            self.nparttypes = nparttypes
            self.coords = coords
        # time and dt will be added along with the whole snapshot
        return time, dt
    
    def add_snapshot (self, filename, verbose=True):
        
        if verbose:
            print(" - adding snapshot %s... " % filename, flush=True, end='')
        
        # Read from the file
        self.file = open(filename, 'rb')
            
        time, dt = self.process_metadata()

        # Read all particle data

        data_type_vector = np.dtype([('x1', np.float32), ('x2', np.float32), ('x3', np.float32)])
        data_type = np.dtype([('pos', data_type_vector), ('vel', data_type_vector), ('dpar', np.float32), ('gr_property', np.int32), ('my_id', np.int64), ('init_id', np.int32)])

        buffer = np.frombuffer(self.file.read(data_type.__sizeof__() *self.npart), dtype=data_type, count=self.npart)

        # organize buffer data (note extra dimension)
        buffer_pos = np.array([[list(x[0]) for x in buffer],])
        buffer_vel = np.array([[list(x[1]) for x in buffer],])
        buffer_dpar = np.array([[x[2] for x in buffer],])
        buffer_grp = np.array([[x[3] for x in buffer],])
        buffer_my_id = np.array([[x[4] for x in buffer],])
        buffer_init_id = np.array([[x[5] for x in buffer],])
        del buffer
        
        self.file.close()
        
        # Add data to the class structures
        if self.initialized: # add time slice
            self.times.append(time)
            self.dts.append(dt)
            self.pos = np.concatenate([self.pos, buffer_pos], axis=0)
            self.vel = np.concatenate([self.vel, buffer_vel], axis=0)
            self.dpar = np.concatenate([self.dpar, buffer_dpar], axis=0)
            self.grp = np.concatenate([self.grp, buffer_grp], axis=0)
            self.my_id = np.concatenate([self.my_id, buffer_my_id], axis=0)
            self.init_id = np.concatenate([self.init_id, buffer_init_id], axis=0)
        else: # initialize
            self.times.append(1.*time)
            self.dts.append(1.*dt)
            self.pos = 1.*buffer_pos
            self.vel = 1.*buffer_vel
            self.dpar = 1.*buffer_dpar
            self.grp = 1.*buffer_grp
            self.my_id = 1.*buffer_my_id
            self.init_id = 1.*buffer_init_id
            self.initialized = True
        del buffer_pos, buffer_vel, buffer_dpar, buffer_grp, buffer_my_id, buffer_init_id
          
        if verbose:
            print('done.', flush=True)

    def sort (self): # sort to id indivitual particles across time
        for idx_t in tqdm(range(len(self.times))):
            sort_idxs = np.argsort(self.my_id[idx_t,:])
            self.pos[idx_t,:,:] = self.pos[idx_t,sort_idxs,:]
            self.vel[idx_t,:,:] = self.vel[idx_t,sort_idxs,:]
            self.dpar[idx_t,:] = self.dpar[idx_t,sort_idxs]
            self.grp[idx_t,:] = self.grp[idx_t,sort_idxs]
            self.my_id[idx_t,:] = self.my_id[idx_t,sort_idxs]
            self.init_id[idx_t,:] = self.init_id[idx_t,sort_idxs]
            
    def update_aux_data (self, to_update=['Ekin',]):
        if 'Ekin' in to_update:
            if self.relativistic:
                self.Ekin = np.sum(1.0 / np.sqrt(1.0 - (self.vel)**2), axis=-1)
            else:
                self.Ekin = 0.5 * np.sum((self.vel)**2, axis=-1)
            
    def plot_pos2D (self, ax, color_by='time', cmap='rainbow', axis_x=0, axis_y=1):
        if color_by == 'time':
            for p in range(self.npart):
                ax.scatter(self.pos[:,p,axis_x], self.pos[:,p,axis_y], s=0.1, c=self.times, cmap=cmap)
        elif color_by == 'particle':
            from matplotlib.cmap import get_cmap
            c = get_cmap(cmap, 128)
            for p in range(self.npart):
                ax.scatter(self.pos[:,p,axis_x], self.pos[:,p,axis_y], s=0.1, color=c(p/self.npart))

        ax.set_aspect(1.0)

        if axis_x == 0:
            ax.set_xlim(self.coords['x1min'],self.coords['x1max'])
            ax.set_xlabel('x')
        elif axis_x == 1:
            ax.set_xlim(self.coords['x2min'],self.coords['x2max'])
            ax.set_xlabel('y')
        elif axis_x == 2:
            ax.set_xlim(self.coords['x3min'],self.coords['x3max'])
            ax.set_xlabel('z')

        if axis_y == 0:
            ax.set_ylim(self.coords['x1min'],self.coords['x1max'])
            ax.set_ylabel('x')
        elif axis_y == 1:
            ax.set_ylim(self.coords['x2min'],self.coords['x2max'])
            ax.set_ylabel('y')
        elif axis_y == 2:
            ax.set_ylim(self.coords['x3min'],self.coords['x3max'])
            ax.set_ylabel('z')

        ax.set_title('Particle position')
                
    def plot_Ekin_vs_time (self, ax, cmap='rainbow', average=False, residuals=False):
        data_to_plot = 1.*self.Ekin[:,:]
        if residuals: # subtract time-average
            data_to_plot -= np.mean(data_to_plot, axis=0)
            ax.set_title('Particle energy: Residuals')
        else:
            ax.set_title('Particle energy')
        if average: # average over particles
            data_to_plot = np.mean(data_to_plot, axis=1)
            ax.scatter(self.times, data_to_plot, s=0.1)
        else: # plot each particle separately
            from matplotlib.cm import get_cmap
            c = get_cmap(cmap, min(512, self.npart))
            for p in range(self.npart):
                ax.scatter(self.times, data_to_plot[:,p], s=0.1, color=c(p/self.npart))
        if residuals:
            ax.set_ylim(1.1*np.min(data_to_plot), 1.1*np.max(data_to_plot))
        ax.set_xlabel('Time')

    def plot_Ekin_distribution (self, ax, cax=None, bins=None, log=True, cmap='rainbow', average=False, i=None, navg=1, data_to_plot=None):

        # generate Ekin if needed
        if len(self.Ekin) == 0:
            self.update_aux_data()

        if average or navg >= len(self.times): # plot time-averaged energy distribution
            data_to_plot = np.mean(self.Ekin, axis=0)
            ax.hist(data_to_plot, bins, density=True, histtype='step', log=log, stacked=False)

        elif i != None: # plot a single frame
            #data_to_plot = np.mean(np.transpose(self.Ekin)[:,i:(i+navg)], axis=1)
            ax.hist(data_to_plot, bins, density=True, histtype='step', log=log, stacked=False)

        else: # plot history of energy distr. color-coded with time
            data_x = np.array(self.times)
            data_y = np.transpose(self.Ekin)
            from matplotlib.cm import get_cmap
            c = get_cmap(cmap, min(512, len(data_x)))
            no_plots = int(len(data_x)/navg)+1
            for idx_t in tqdm(range(no_plots)):
                data_to_plot = np.mean(data_y[:,(navg*idx_t):min(len(data_x),navg*(idx_t+1))], axis=1)
                ax.hist(data_to_plot, bins, density=True, histtype='step', log=log, stacked=False, color=c(idx_t/no_plots))
            if cax != None: # add a color bar
                from matplotlib.pyplot import colorbar
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                colorbar(ScalarMappable(Normalize(vmin=data_x[0], vmax=data_x[-1]), cmap=c), cax=cax)
                cax.set_ylabel('Time [sim.u.]')

        ax.set_ylabel('PDF')
        ax.set_xlabel('Particle energy')

    def _plot_Ekin_distribution_frame (self, data_to_plot, bins=None, log=True, cmap='rainbow', i=None, navg=1, tempdir='./temp_EkinDistr/', force=False, verbose=True, xmin=None, xmax=None, ymin=None, ymax=None):
        filename = ('EkinDistr_%08i.png' % i)
        if os.path.isfile(tempdir + filename) and not force:
            if verbose:
                print(' - %s already processed. Skipping.' % filename, flush=True)
            pass
        else:
            if verbose:
                print(' - processing %s..' % filename, flush=True)
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=[8,6])
            ax = plt.gca()
            self.plot_Ekin_distribution(ax, bins=bins, log=log, cmap=cmap, i=i, navg=navg, data_to_plot=data_to_plot['data'][:,i])
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            ax.set_title('Ekin distribution, $t = %.2f$ sim.u.' % (data_to_plot['times'][i],))
            fig.savefig(tempdir + filename, format='png')
            plt.close(fig)

    # WARNING: movie plots will not work in parallel if matplotlib is imported before calling this function!
    # [if you know how to fix this, please let me know ;) ]
    def plot_Ekin_distribution_movie (self, nproc=1, bins=None, log=True, cmap='rainbow', navg=1, tempdir='./temp_EkinDistr/', outdir='./', force=False, verbose=True, xmin=None, xmax=None, ymin=None, ymax=None):
        # create the temp folder if needed
        if not os.path.exists(tempdir):
            if verbose:
                print("Preparing tempdir..", end='', flush=True)
            os.makedirs(tempdir)
            if verbose:
                print(" done", flush=True)
        # generate Ekin if needed
        if len(self.Ekin) == 0:
            if verbose:
                print("Calculating Ekin..", end='', flush=True)
            self.update_aux_data()
            if verbose:
                print(" done", flush=True)
        # boxcar-average before plotting for more efficient code
        data_to_plot = {'times': 1.*np.array(self.times), 'data': np.transpose(self.Ekin)}
        if navg > 1:
            if verbose:
                print("Boxcar-averaging..", end='', flush=True)
            boxcar = np.ones(navg) * 1. / navg
            data_to_plot['times'] = sp_convolve(data_to_plot['times'], boxcar, mode='valid')
            boxcar.shape = (1,navg)
            data_to_plot['data'] = sp_convolve(data_to_plot['data'], boxcar, mode='valid')
            if verbose:
                print(" done", flush=True)
        # plot the frames (ideally in parallel)
        if verbose:
            print("Generating frames..", flush=True)
        if nproc == 1:
            for i in range(len(self.times)-navg):
                self.plot_Ekin_distribution_frame(data_to_plot, bins=bins, log=log, cmap=cmap, i=i, navg=navg, tempdir=tempdir, force=force, verbose=verbose, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        else:
            with ProcessPool(nproc) as pool:
                _ = pool.map(lambda i : self._plot_Ekin_distribution_frame(data_to_plot, bins=bins, log=log, cmap=cmap, i=i, navg=navg, tempdir=tempdir, force=force, verbose=verbose, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), range(len(self.times)-navg))
        if verbose:
            print(" All frames generated.", flush=True)
        # render the movie
        try:
            print("Rendering the movie..", flush=True)
            command = ("ffmpeg -y -r 20 -f image2 -i \"%sEkinDistr_%%*.png\" -f mp4 -q:v 0 -vcodec mpeg4 -r 20 %sEkinDistr.mp4" % (tempdir, outdir))
            print(command, flush=True)
            os.system(command)
        except Exception as e:
            print('Error while rendering movie:\n%s\n -- please try to manually convert the .png files generated in %s.' % (e, tempdir), flush=True)
        # clean up
        del data_to_plot
        print('Done.', flush=True)