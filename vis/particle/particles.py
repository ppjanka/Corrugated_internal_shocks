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

import cartographic as cart

# find all elements in a that do not appear in b
def array_diff (a,b):
    return np.array(list(set([tuple(x) for x in a]).difference(set([tuple(x) for x in b]))))

class Particles:
    
    def __init__ (self, relativistic=False): # empty class
        self.initialized = False
        self.sorted = True
        self.relativistic = relativistic
        self.npart = 0
        self.nparttypes = 0
        self.coords = {}
        self.times = []
        self.dts = []
        self.uniq_id = [] # (my_id, init_id) pair for each particle in the database (rows filled with np.nan if not existent at a given time step)
        # auxiliary data calculated from the above
        # arrays with particle data:
        #   [ dimensions: <time> <particle> <quantity> ]
        self.pos = []
        self.vel = []
        self.dpar = []
        self.grp = []
        #self.my_id = []
        #self.init_id = []
        self.shock_of_origin = []
        self.injected = []
        self.Ekin = []
        self.vel_theta = []
        self.vel_phi = []
        self.hist = {}
        
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
            if self.coords != coords: #self.npart != npart or self.nparttypes != nparttypes
                print('ERROR: metadata mismatch, aborting.', flush=True)
                print(' self.npart = %i, npart = %i,' % (self.npart, npart))
                print(' self.nparttypes = %i, nparttypes = %i,' % (self.nparttypes, nparttypes))
                print(' self.coords = %s\n coords = %s.' % (self.coords, coords), flush=True)
                sys.exit(1)
        else: # initialize metadata
            self.npart = npart
            self.nparttypes = nparttypes
            self.coords = coords
        # time and dt will be added along with the whole snapshot
        return time, dt, npart

    def get_time (self, filename):
        # Read from the file
        self.file = open(filename, 'rb')
        return self.process_metadata()[:2]

    def drop_first_snapshot (self, n_to_drop=1):
        if initialized:
            n = min(len(self.times)-1, n_to_drop)
            # raw data
            self.times = self.times[n:]
            self.dts = self.dts[n:]
            self.pos = self.pos[n:]
            self.vel = self.vel[n:]
            self.dpar = self.dpar[n:]
            self.grp = self.grp[n:]
            #self.my_id = self.my_id[n:]
            #self.init_id = self.init_id[n:]
            self.uniq_id = self.uniq_id[n:]
            # auxiliary data
            if len(self.Ekin) > 0:
                self.Ekin = self.Ekin[n:]
            if len(self.val_theta) > 0:
                self.vel_theta = self.vel_theta[n:]
            if len(self.vel_phi) > 0:
                self.vel_phi = self.vel_phi[n:]
            self.hist = {}
    
    def add_snapshot (self, filename, verbose=True):
        
        if verbose:
            print(" - adding snapshot %s... " % filename, flush=True, end='')
        
        # Read from the file
        self.file = open(filename, 'rb')
            
        time, dt, npart = self.process_metadata()

        # Read all particle data

        data_type_vector = np.dtype([('x1', np.float32), ('x2', np.float32), ('x3', np.float32)])
        data_type = np.dtype([('pos', data_type_vector), ('vel', data_type_vector), ('dpar', np.float32), ('gr_property', np.int32), ('my_id', np.int64), ('init_id', np.int32), ('shock_of_origin', np.int16), ('injected', np.int16)])

        buffer = np.frombuffer(self.file.read(data_type.__sizeof__() *self.npart), dtype=data_type, count=npart)

        # organize buffer data (note extra dimension)
        buffer_pos = np.array([[list(x[0]) for x in buffer],])
        buffer_vel = np.array([[list(x[1]) for x in buffer],])
        buffer_dpar = np.array([[x[2] for x in buffer],])
        buffer_grp = np.array([[x[3] for x in buffer],])
        #buffer_my_id = np.array([[x[4] for x in buffer],])
        #buffer_init_id = np.array([[x[5] for x in buffer],])
        buffer_shock_of_origin = np.array([[x[6] for x in buffer],])
        buffer_injected = np.array([[x[7] for x in buffer],])

        # read in the unique particle ids
        uniq_id, idxs = np.unique(np.array([ [x[5],x[4]] for x in buffer]).astype(np.int), axis=0, return_index=True)

        # np.unique sorts the data, ensure the same order
        buffer_pos = buffer_pos[:,idxs]
        buffer_vel = buffer_vel[:,idxs]
        buffer_dpar = buffer_dpar[:,idxs]
        buffer_grp = buffer_grp[:,idxs]
        #buffer_my_id = buffer_my_id[:,idxs]
        #buffer_init_id = buffer_init_id[:,idxs]
        buffer_shock_of_origin = buffer_shock_of_origin[:,idxs]
        buffer_injected = buffer_injected[:,idxs]

        if self.initialized:
            # compare uniq_id in the class with uniq_id in the file
            add_to_self = array_diff(uniq_id, self.uniq_id[-1])
            add_to_file = array_diff(self.uniq_id[-1], uniq_id)
            # do we need to add any uniq_id to the class?
            if len(add_to_self) > 0:
                if verbose:
                    print('   > adding %i particles to the class', len(add_to_self), flush=True)
                # add np.nan to particle data where it did not exist
                buffer_nan = np.ones((self.pos.shape[0],len(add_to_self),3)) * np.nan
                self.pos = np.concatenate([self.pos, 1.*buffer_nan], axis=1)
                self.vel = np.concatenate([self.vel, 1.*buffer_nan], axis=1)
                buffer_nan = np.ones((self.pos.shape[0],len(add_to_self),3)) * np.nan
                self.dpar = np.concatenate([self.dpar, 1.*buffer_nan], axis=1)
                self.grp = np.concatenate([self.grp, 1.*buffer_nan], axis=1)
                #self.my_id = np.concatenate([self.my_id, 1.*buffer_nan], axis=1)
                #self.init_id = np.concatenate([self.init_id, 1.*buffer_nan], axis=1)
                self.shock_of_origin = np.concatenate([self.shock_of_origin, 1.*buffer_nan], axis=1)
                self.injected = np.concatenate([self.injected, 1.*buffer_nan], axis=1)
                # auxiliary data
                if len(self.Ekin) > 0:
                    self.Ekin = np.concatenate([self.Ekin, 1.*buffer_nan], axis=1)
                if len(self.val_theta) > 0:
                    self.vel_theta = np.concatenate([self.vel_theta, 1.*buffer_nan], axis=1)
                if len(self.vel_phi) > 0:
                    self.vel_phi = np.concatenate([self.vel_phi, 1.*buffer_nan], axis=1)
                buffer_nan = np.tile(add_to_self, (self.pos.shape[0],1))
                self.uniq_id = np.concatenate([self.uniq_id, buffer_nan], axis=1)
                del buffer_nan
                self.hist = {}
                # update the particle number
                self.npart = self.uniq_id.shape[1]
            # do we need to add any uniq_id to the data from the file?
            if len(add_to_file) > 0:
                if verbose:
                    print('   > adding %i particles to the file data', len(add_to_file), flush=True)
                uniq_id = np.concatenate([uniq_id, add_to_file], axis=0)
                # add np.nan to particle data where it no longer exists
                buffer_nan = np.ones((1,len(add_to_file),3)) * np.nan
                buffer_pos = np.concatenate([buffer_pos, 1.*buffer_nan], axis=1)
                buffer_vel = np.concatenate([buffer_vel, 1.*buffer_nan], axis=1)
                buffer_nan = np.ones((1,len(add_to_file))) * np.nan
                buffer_dpar = np.concatenate([buffer_dpar, 1.*buffer_nan], axis=1)
                buffer_grp = np.concatenate([buffer_grp, 1.*buffer_nan], axis=1)
                #buffer_my_id = np.concatenate([buffer_my_id, 1.*buffer_nan], axis=1)
                #buffer_init_id = np.concatenate([buffer_init_id, 1.*buffer_nan], axis=1)
                buffer_shock_of_origin = np.concatenate([buffer_shock_of_origin, 1.*buffer_nan], axis=1)
                buffer_injected = np.concatenate([buffer_injected, 1.*buffer_nan], axis=1)
                del buffer_nan
            # final check
            if len(uniq_id) != len(self.uniq_id[-1]):
                print('ERROR: uniq_id mismatch between the new file and Particles. Could not resolve. Aborting.', flush=True)
                sys.exit()

        # clean up
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
            #self.my_id = np.concatenate([self.my_id, buffer_my_id], axis=0)
            #self.init_id = np.concatenate([self.init_id, buffer_init_id], axis=0)
            self.uniq_id = np.concatenate([self.uniq_id, [uniq_id,]], axis=0)
            self.shock_of_origin = np.concatenate([self.shock_of_origin, buffer_shock_of_origin], axis=0)
            self.injected = np.concatenate([self.injected, buffer_injected], axis=0)
        else: # initialize
            self.times.append(1.*time)
            self.dts.append(1.*dt)
            self.pos = 1.*buffer_pos
            self.vel = 1.*buffer_vel
            self.dpar = 1.*buffer_dpar
            self.grp = 1.*buffer_grp
            #self.my_id = 1.*buffer_my_id
            #self.init_id = 1.*buffer_init_id
            self.uniq_id = np.array([uniq_id,])
            self.shock_of_origin = 1.*buffer_shock_of_origin
            self.injected = 1.*buffer_injected
            self.initialized = True
        del buffer_pos, buffer_vel, buffer_dpar, buffer_grp, uniq_id, buffer_shock_of_origin, buffer_injected #, buffer_my_id, buffer_init_id

        self.sorted = False
          
        if verbose:
            print('done.', flush=True)

    def add_snapshot_FIFO (self, filename, verbose=True):
        # drop the oldest snapshot
        self.drop_first_snapshot()
        # add the new snapshot
        self.add_snapshot(filename, verbose)

    def sort (self, verbose=False): # sort to id indivitual particles across time
        if not self.sorted:
            if verbose:
                print('Sorting particles..', flush=True)
                t = tqdm
            else:
                t = lambda x : x
            for idx_t in t(range(len(self.times))):
                # sort first by init_id, then by my_id (reverse order with argsort)
                dtype_uniq = np.dtype([('init_id', np.int32), ('my_id', np.int32)])
                sort_idxs = self.uniq_id[idx_t,:,:].astype(np.int32).view(dtype_uniq).argsort(order=['init_id','my_id'], axis=0)[:,0]
                self.pos[idx_t,:,:] = self.pos[idx_t,sort_idxs,:]
                self.vel[idx_t,:,:] = self.vel[idx_t,sort_idxs,:]
                self.dpar[idx_t,:] = self.dpar[idx_t,sort_idxs]
                self.grp[idx_t,:] = self.grp[idx_t,sort_idxs]
                #self.my_id[idx_t,:] = self.my_id[idx_t,sort_idxs]
                #self.init_id[idx_t,:] = self.init_id[idx_t,sort_idxs]
                self.uniq_id[idx_t,:] = self.uniq_id[idx_t,sort_idxs]
                self.shock_of_origin[idx_t,:] = self.shock_of_origin[idx_t,sort_idxs]
                self.injected[idx_t,:] = self.injected[idx_t,sort_idxs]
            self.sorted = True
            
    # TODO: only update Ekin that has not been calculated before
    def update_aux_data (self, to_update=['Ekin',]):
        if 'Ekin' in to_update:
            if self.relativistic:
                self.Ekin = np.sum(1.0 / np.sqrt(1.0 - (self.vel)**2), axis=-1)
            else:
                self.Ekin = 0.5 * np.sum((self.vel)**2, axis=-1)

    # returns n indices of particles with highest Ekin at the final or ith frame
    def find_highest_Ekin (self, n=1, i=None):

        if i == None:
            idx = len(self.times)-1
        else:
            idx = i

        # generate Ekin if needed
        if len(self.Ekin) != len(self.uniq_id):
            self.update_aux_data(to_update=['Ekin',])

        # count np.nan values to ignore (they will show up as the highest energies)
        nan_count = np.isnan(self.Ekin[idx,:]).sum()

        # find ids of the particles with n highest energies
        return self.uniq_id[idx, np.argpartition(self.Ekin[idx,:],-n-nan_count)[(-n-nan_count):(-nan_count)]]

    # returns n indices of particles with Ekin closest to median at the final or ith frame
    def find_median_Ekin (self, n=1, i=None):

        if i == None:
            idx = len(self.times)-1
        else:
            idx = i

        # generate Ekin if needed
        if len(self.Ekin) != len(self.uniq_id):
            self.update_aux_data(to_update=['Ekin',])

        # count np.nan values to ignore (they will show up as the highest energies)
        nan_count = np.isnan(self.Ekin[idx,:]).sum()

        # find ids:
        part_idx = int(0.5*(self.npart+n))
        return self.uniq_id[idx, np.argpartition(self.Ekin[idx,:],kth=[-part_idx-nan_count,part_idx-nan_count])[(-part_idx-nan_count):(part_idx+1-nan_count)]]
            
    def plot_pos2D (self, ax, color_by='time', cmap='rainbow', axis_x=0, axis_y=1, annotations=True, selection=[]):

        # choose coloring
        if color_by == 'time':
            from matplotlib.cm import get_cmap
            cmap_here = get_cmap(cmap, 128)
            tmin, tmax = min(self.times), max(self.times)
            c = lambda x : cmap_here((np.array(self.times)-tmin)/(tmax-tmin))
        elif color_by == 'particle':
            from matplotlib.cm import get_cmap
            c = get_cmap(cmap, 128)
            # sort if needed
            self.sort(verbose=True)
        else: #solid color assumed
            c = lambda x : color_by

        if len(selection) > 0:
            p_idxs = np.array(selection).astype(np.int64)
        else:
            p_idxs = range(self.npart)

        # plot the traces
        for p in p_idxs:
            ax.scatter(self.pos[:,p,axis_x], self.pos[:,p,axis_y], s=0.1, color=c(p/self.npart))

        ax.set_aspect(1.0)

        if annotations:
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
            ax.set_ylim(1.1*np.nanmin(data_to_plot), 1.1*np.nanmax(data_to_plot))
        ax.set_xlabel('Time')

    def plot_Ekin_distribution (self, ax, cax=None, bins=None, log=True, cmap='rainbow', i=None, navg=None, history=False, data_to_plot=None):

        # generate Ekin if needed
        if len(self.Ekin) != len(self.uniq_id):
            self.update_aux_data(to_update=['Ekin',])

        if not history:

            if i == None: # plot time-averaged energy distribution
                data_to_plot = np.mean(self.Ekin, axis=0)
                ax.hist(data_to_plot, bins, density=True, histtype='step', log=log, stacked=False)

            elif navg == None: # plot a single frame
                ax.hist(data_to_plot, bins, density=True, histtype='step', log=log, stacked=False)

            else: # plot an average from the frames [i-navg, i]
                data_to_plot = np.mean(self.Ekin[(i-navg):(i+1),:], axis=0)
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
        if len(self.Ekin) != len(self.uniq_id):
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

    def plot_direction_distribution_3D (self, ax, projection, res, i=None, navg=None, cmap='rainbow', res_patch=8, cax=None, weights=None, recalculate=True):

        name = 'direction3D'
        if weights != None:
            name += "_" + weights

        # first, create an angular grid for direction bins, with equal-area components
        thetas = np.arccos(np.linspace(1.,-1., res))
        phis = np.linspace(-np.pi,np.pi, 2*res)

        if recalculate or (name not in self.hist.keys()):

            if i == None: # calculate statistics for the entire simulation
                vel = self.vel
            elif navg == None: # print a single frame
                vel = self.vel[i,:,:].reshape(1,self.npart,3)
            else: # plot the population from the frames [i-navg, i]
                vel = self.vel[(i-navg):(i+1),:,:]

            # calculate angles for the particles
            if recalculate or len(self.vel_theta) == 0 or len(self.vel_phi) == 0:
                vel_xy = np.sqrt(vel[:,:,0]**2+vel[:,:,1]**2)
                self.vel_theta = np.arctan(vel_xy/vel[:,:,2])
                self.vel_theta = np.where(self.vel_theta > 0, self.vel_theta, self.vel_theta + np.pi)
                self.vel_phi = np.arccos(vel[:,:,0]/vel_xy)
                self.vel_phi = np.where(vel[:,:,1] > 0, self.vel_phi, 2.*np.pi - self.vel_phi)
                self.vel_phi = np.where(self.vel_phi < np.pi, self.vel_phi, self.vel_phi-2.*np.pi) # move to the [-pi,pi] range
                self.vel_theta = self.vel_theta.flatten()
                self.vel_phi = self.vel_phi.flatten()
            if weights == 'mom':
                weights = np.sqrt(np.sum(vel**2,axis=2)).flatten()

            # calculate the 2d histogram
            # (values, theta_edges, phi_edges)
            self.hist[name] = np.histogram2d(self.vel_theta, self.vel_phi, bins=[thetas, phis], density=False, weights=weights)

        # plot the 2d histogram
        cart.plot_projected(ax, projection, thetas, phis, self.hist[name][0], in_unit='rad', cmap=cmap, res_patch=res_patch, cax=cax)

        # plot the grid
        cart.plot_cartographic_grid(ax, projection, phi_ticks=[-90.,0.,90.], theta_ticks=[30.,60.,90.,120.,150.])

    def plot_direction_distribution_2D (self, ax, res, i=None, weights=None, recalculate=True):

        name = 'direction2D'
        if weights != None:
            name += "_" + weights
        # first, create an angular grid for direction bins, with equal-area components
        phis = np.linspace(-np.pi,np.pi, 2*res)

        if recalculate or (name not in self.hist.keys()):

            # recalculate from the 3D histograms?
            name_3D = 'direction3D'
            if weights != None:
                name_3D += "_" + weights
            if not recalculate and (name_3D in self.hist.keys()):

                # (values, phi_edges)
                self.hist[name] = ( np.sum(self.hist[name_3D][0], axis=0), self.hist[name_3D][2] )

            else: # calculate from scratch

                if i != None: # print a single frame
                    vel = self.vel[i,:,:].reshape(1,self.npart,3)
                else: # calculate statistics for the entire simulation
                    vel = self.vel

                # calculate angles for the particles
                if recalculate or len(self.vel_phi) == 0:
                    vel_xy = np.sqrt(vel[:,:,0]**2+vel[:,:,1]**2)
                    self.vel_phi = np.arccos(vel[:,:,0]/vel_xy)
                    self.vel_phi = np.where(vel[:,:,1] > 0, self.vel_phi, 2.*np.pi - self.vel_phi)
                    self.vel_phi = np.where(self.vel_phi < np.pi, self.vel_phi, self.vel_phi-2.*np.pi) # move to the [-pi,pi] range
                    self.vel_phi = self.vel_phi.flatten()
                if weights == 'mom':
                    weights = np.sqrt(np.sum(vel**2,axis=2)).flatten()

                # calculate the 2d histogram
                # (values, theta_edges, phi_edges)
                self.hist[name] = np.histogram(self.vel_phi, bins=phis, density=False, weights=weights)

            # plot
            ax.set_aspect(1.0)
            datax = np.repeat(phis,2)
            datay = np.repeat(self.hist[name][0],2)
            datay = np.insert(datay, [0,len(datay)], [datay[0],datay[0]])
            ax.plot(datax, datay)