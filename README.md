# Athena-Cversion
This is a fork of the publicly available Athena-Cversion MHD code (hereafter, Athena 4.2, please see relevant references in https://princetonuniversity.github.io/Athena-Cversion/AthenaDocsMP). For full documentation of the original code please go to https://princetonuniversity.github.io/Athena-Cversion/.

Athena 4.2 includes a particle module intended for simulations of dust granules in protoplanetary disks (designed by Bai & Stone, 2010). In the original module, the particles are assumed to be non-relativistic and are coupled to the fluid by drag forces.

Here, I have used the module of Bai & Stone (2010) as a basis (utilizing the MPI handling of particles implemented by them), and extended it to handle charged relativistic test particles coupled to the fluid only by Lorentz forces (one can think of them as "cosmic rays"). Note that no backreaction on the fluid is implemented (i.e., it is assumed that the test particles are not dynamically important for behavior of the fluid). I have also implemented additional diagnostics to thus modified particle module, written in Python.

Added features:
 - charged relativistic trace / test particles in cartesian coordinates,
 - a number of Python diagnostic scripts to be used with thus modified particle module.
