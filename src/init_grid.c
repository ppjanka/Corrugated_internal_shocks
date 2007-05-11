#include "copyright.h"
/*==============================================================================
 * FILE: init_grid.c
 *
 * PURPOSE: Initializes most variables in the Grid structure:
 *      time,nstep,[ijk]s,[ijk]e,dx[123],[ijk]disp,x[123]_0
 *   Allocates memory for gas arrays and interface B.
 *
 * The Grid may be just one block in the Domain (for MPI parallel jobs), or the
 * entire Domain (for single-processor jobs).  The number of cells in, and
 * location of this Grid in the Domain, is determined by init_domain(), which
 * should be called before this routine (even for single processor jobs).
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   init_grid()
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "prototypes.h"

void init_grid(Grid *pG, Domain *pD)
{
  int Nx1T,Nx2T,Nx3T;    /* Total Number of grid cells in x1,x2,x3 direction */
  int ib,jb,kb;
  Real x1min,x1max,x2min,x2max,x3min,x3max;   /* read from input file */

/* initialize time, nstep */

  pG->time = 0.0;
  pG->nstep = 0;

/* get (i,j,k) coordinates of grid being updated on this processor */

  get_myblock_ijk(pD, pG->my_id, &ib, &jb, &kb);

/* ---------------------  Intialize grid in 1-direction --------------------- */
/* Initialize is,ie */

  pG->Nx1 = pD->grid_block[kb][jb][ib].ixe - pD->grid_block[kb][jb][ib].ixs + 1;

  if(pG->Nx1 > 1) {
    pG->is = nghost;
    pG->ie = pG->Nx1 + nghost - 1;
  }
  else
    pG->is = pG->ie = 0;

/* Compute dx1 */

  x1min = par_getd("grid","x1min");
  x1max = par_getd("grid","x1max");
  if(x1max < x1min) {
    ath_error("[init_grid]: x1max = %g < x1min = %g\n",x1max,x1min);
  }

  pG->dx1 = (x1max - x1min)/(Real)(pD->ixe - pD->ixs + 1);

/* Initialize i-displacement, and the x1-position of coordinate ix = 0. */

  pG->idisp = pD->grid_block[kb][jb][ib].ixs - pG->is;
  pG->x1_0 = x1min; 

/* ---------------------  Intialize grid in 2-direction --------------------- */
/* Initialize js,je */

  pG->Nx2 = pD->grid_block[kb][jb][ib].jxe - pD->grid_block[kb][jb][ib].jxs + 1;

  if(pG->Nx2 > 1) {
    pG->js = nghost;
    pG->je = pG->Nx2 + nghost - 1;
  }
  else
    pG->js = pG->je = 0;

/* Compute dx2 */

  x2min = par_getd("grid","x2min");
  x2max = par_getd("grid","x2max");
  if(x2max < x2min) {
    ath_error("[init_grid]: x2max = %g < x2min = %g\n",x2max,x2min);
  }
  pG->dx2 = (x2max - x2min)/(Real)(pD->jxe - pD->jxs + 1);

/* Initialize j-displacement, and the x2-position of coordinate jx = 0. */

  pG->jdisp = pD->grid_block[kb][jb][ib].jxs - pG->js;
  pG->x2_0 = x2min;

/* ---------------------  Intialize grid in 3-direction --------------------- */
/* Initialize ks,ke */

  pG->Nx3 = pD->grid_block[kb][jb][ib].kxe - pD->grid_block[kb][jb][ib].kxs + 1;

  if(pG->Nx3 > 1) {
    pG->ks = nghost;
    pG->ke = pG->Nx3 + nghost - 1;
  }
  else
    pG->ks = pG->ke = 0;

/* Compute dx3 */

  x3min = par_getd("grid","x3min");
  x3max = par_getd("grid","x3max");
  if(x3max < x3min) {
    ath_error("[init_grid]: x3max = %g < x3min = %g\n",x3max,x3min);
  }
  pG->dx3 = (x3max - x3min)/(Real)(pD->kxe - pD->kxs + 1);

/* Initialize k-displacement, and the x3-position of coordinate kx = 0. */

  pG->kdisp = pD->grid_block[kb][jb][ib].kxs - pG->ks;
  pG->x3_0 = x3min;

/* ---------  Allocate 3D arrays to hold Gas based on size of grid --------- */

  if (pG->Nx1 > 1)
    Nx1T = pG->Nx1 + 2*nghost;
  else
    Nx1T = 1;

  if (pG->Nx2 > 1)
    Nx2T = pG->Nx2 + 2*nghost;
  else
    Nx2T = 1;

  if (pG->Nx3 > 1)
    Nx3T = pG->Nx3 + 2*nghost;
  else
    Nx3T = 1;

/* Build a 3D array of type Gas */

  pG->U = (Gas***)calloc_3d_array(Nx3T, Nx2T, Nx1T, sizeof(Gas));
  if (pG->U == NULL) goto on_error;

/* Build 3D arrays to hold interface field */

#ifdef MHD
  pG->B1i = (Real***)calloc_3d_array(Nx3T, Nx2T, Nx1T, sizeof(Real));
  if (pG->B1i == NULL) {
    free_3d_array(pG->U);
    goto on_error;
  }

  pG->B2i = (Real***)calloc_3d_array(Nx3T, Nx2T, Nx1T, sizeof(Real));
  if (pG->B2i == NULL) {
    free_3d_array(pG->U);
    free_3d_array(pG->B1i);
    goto on_error;
  }

  pG->B3i = (Real***)calloc_3d_array(Nx3T, Nx2T, Nx1T, sizeof(Real));
  if (pG->B3i == NULL) {
    free_3d_array(pG->U);
    free_3d_array(pG->B1i);
    free_3d_array(pG->B2i);
    goto on_error;
  }
#endif /* MHD */

  return;

  on_error:
    ath_error("[init_grid]: Error allocating memory\n");
}
