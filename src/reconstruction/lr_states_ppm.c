#include "../copyright.h"
/*==============================================================================
 * FILE: lr_states_ppm.c
 *
 * PURPOSE: Third order (piecewise parabolic) spatial reconstruction using
 *   characteristic interpolation in the primitive variables.  A time-evolution
 *   (characteristic tracing) step is used to interpolate interface values to
 *   the half time level {n+1/2}, unless the unsplit integrator in 3D is VL.
 *
 * NOTATION:
 *   W_{L,i-1/2} is reconstructed value on the left-side of interface at i-1/2
 *   W_{R,i-1/2} is reconstructed value on the right-side of interface at i-1/2
 *
 *   The L- and R-states at the left-interface in each cell are indexed i.
 *   W_{L,i-1/2} is denoted by Wl[i  ];   W_{R,i-1/2} is denoted by Wr[i  ]
 *   W_{L,i+1/2} is denoted by Wl[i+1];   W_{R,i+1/2} is denoted by Wr[i+1]
 *
 *   Internally, in this routine, Wlv and Wrv are the reconstructed values on
 *   the left-and right-side of cell center.  Thus (see Step 18),
 *     W_{L,i-1/2} = Wrv(i-1);  W_{R,i-1/2} = Wlv(i)
 *
 * REFERENCE:
 *   P. Colella & P. Woodward, "The piecewise parabolic method (PPM) for
 *   gas-dynamical simulations", JCP, 54, 174 (1984).
 *
 * CONTAINS PUBLIC FUNCTIONS:
 *   lr_states()          - computes L/R states
 *   lr_states_init()     - initializes memory for static global arrays
 *   lr_states_destruct() - frees memory for static global arrays
 *============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "prototypes.h"
#include "../prototypes.h"

#ifdef THIRD_ORDER

static Real **pW=NULL, **dWm=NULL, **Wim1h=NULL;

/*----------------------------------------------------------------------------*/
/* lr_states:
 * Input Arguments:
 *   W = PRIMITIVE variables at cell centers along 1-D slice
 *   Bxc = B in direction of slice at cell center
 *   dt = timestep;  dtodx = dt/dx
 *   il,iu = lower and upper indices of zone centers in slice
 * W and Bxc must be initialized over [il-3:iu+3]
 *
 * Output Arguments:
 *   Wl,Wr = L/R-states of PRIMITIVE variables at interfaces over [il:iu+1]
 */

void lr_states(const Prim1D W[], MHDARG( const Real Bxc[] , )
	       const Real dt, const Real dtodx, const int il, const int iu,
	       Prim1D Wl[], Prim1D Wr[])
{
  int i,n,m;
  Real lim_slope1,lim_slope2,qa,qb,qc,qx;
  Real ev    [NWAVE],rem    [NWAVE][NWAVE],lem    [NWAVE][NWAVE];
  Real ev_ip1[NWAVE],rem_ip1[NWAVE][NWAVE],lem_ip1[NWAVE][NWAVE];
  Real dWc[NWAVE+NSCALARS],dWl[NWAVE+NSCALARS];
  Real dWr[NWAVE+NSCALARS],dWg[NWAVE+NSCALARS];
  Real dac[NWAVE+NSCALARS],dal[NWAVE+NSCALARS];
  Real dar[NWAVE+NSCALARS],dag[NWAVE+NSCALARS],da[NWAVE+NSCALARS];
  Real Wlv[NWAVE+NSCALARS],Wrv[NWAVE+NSCALARS];
  Real dW[NWAVE+NSCALARS],W6[NWAVE+NSCALARS];
  Real *pWl, *pWr;

/* Zero eigenmatrices, set pointer to primitive variables */
  for (n=0; n<NWAVE; n++) {
    for (m=0; m<NWAVE; m++) {
      rem[n][m] = 0.0;
      lem[n][m] = 0.0;
      rem_ip1[n][m] = 0.0;
      lem_ip1[n][m] = 0.0;
    }
  }
  for (i=il-3; i<=iu+3; i++) pW[i] = (Real*)&(W[i]);

/*====================== START LOOP OVER il-2:il-1 ==========================*/

/*--- Step 1. ------------------------------------------------------------------
 * Compute eigensystem in primitive variables.  */

  for (i=il-2; i<=il-1; i++) {

#ifdef HYDRO
#ifdef ISOTHERMAL
    esys_prim_iso_hyd(W[i].d,W[i].Vx,       ev,rem,lem);
#else
    esys_prim_adb_hyd(W[i].d,W[i].Vx,Gamma*W[i].P,ev,rem,lem);
#endif /* ISOTHERMAL */
#endif /* HYDRO */

#ifdef MHD
#ifdef ISOTHERMAL
    esys_prim_iso_mhd(W[i].d,W[i].Vx,       Bxc[i],W[i].By,W[i].Bz,ev,rem,lem);
#else
    esys_prim_adb_mhd(W[i].d,W[i].Vx,Gamma*W[i].P,Bxc[i],W[i].By,W[i].Bz,ev,rem,lem);
#endif /* ISOTHERMAL */
#endif /* MHD */

/*--- Step 2. ------------------------------------------------------------------
 * Compute centered, L/R, and van Leer differences of primitive variables
 * Note we access contiguous array elements by indexing pointers for speed */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      dWc[n] = pW[i+1][n] - pW[i-1][n];
      dWl[n] = pW[i][n]   - pW[i-1][n];
      dWr[n] = pW[i+1][n] - pW[i][n];
      if (dWl[n]*dWr[n] > 0.0) {
        dWg[n] = 2.0*dWl[n]*dWr[n]/(dWl[n]+dWr[n]);
      } else {
        dWg[n] = 0.0;
      }
    }

/*--- Step 3. ------------------------------------------------------------------
 * Project differences in primitive variables along characteristics */

    for (n=0; n<NWAVE; n++) {
      dac[n] = lem[n][0]*dWc[0];
      dal[n] = lem[n][0]*dWl[0];
      dar[n] = lem[n][0]*dWr[0];
      dag[n] = lem[n][0]*dWg[0];
      for (m=1; m<NWAVE; m++) {
	dac[n] += lem[n][m]*dWc[m];
	dal[n] += lem[n][m]*dWl[m];
	dar[n] += lem[n][m]*dWr[m];
	dag[n] += lem[n][m]*dWg[m];
      }
    }

/* Advected variables are treated differently; for them the right and left
 * eigenmatrices are simply the identitiy matrix.
 */
#if (NSCALARS > 0)
    for (n=NWAVE; n<(NWAVE+NSCALARS); n++) {
      dac[n] = dWc[n];
      dal[n] = dWl[n];
      dar[n] = dWr[n];
      dag[n] = dWg[n];
    }
#endif

/*--- Step 4. ------------------------------------------------------------------
 * Apply monotonicity constraints to characteristic projections */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      da[n] = 0.0;
      if (dal[n]*dar[n] > 0.0) {
        lim_slope1 = MIN(    fabs(dal[n]),fabs(dar[n]));
        lim_slope2 = MIN(0.5*fabs(dac[n]),fabs(dag[n]));
        da[n] = SIGN(dac[n])*MIN(2.0*lim_slope1,lim_slope2);
      }
    }

/*--- Step 5. ------------------------------------------------------------------
 * Project monotonic slopes in characteristic back to primitive variables  */

    for (n=0; n<NWAVE; n++) {
      dWm[i][n] = da[0]*rem[n][0];
      for (m=1; m<NWAVE; m++) {
        dWm[i][n] += da[m]*rem[n][m];
      }
    }

#if (NSCALARS > 0)
    for (n=NWAVE; n<(NWAVE+NSCALARS); n++) {
      dWm[i][n] = da[n];
    }
#endif

/*--- Step 6. ------------------------------------------------------------------
 * Limit velocity difference to sound speed
 * Limit velocity so momentum is always TVD (using only minmod limiter)
 * CURRENTLY NOT USED.  Was added to make code more robust for turbulence
 * simulations, but found it added noise to Noh shocktube.
 */ 

#ifdef H_CORRECTION
/* 
#ifdef ISOTHERMAL
    qa = Iso_csound;
#else
    qa = sqrt(Gamma*W[i].P/W[i].d);
#endif
    dWm[i][1] = SIGN(dWm[i][1])*MIN(fabs(dWm[i][1]),qa);
*/
#endif /* H_CORRECTION */
/* 
    qa = W[i  ].Vx*W[i  ].d - W[i-1].Vx*W[i-1].d;
    qb = W[i+1].Vx*W[i+1].d - W[i  ].Vx*W[i  ].d;
    qc = W[i+1].Vx*W[i+1].d - W[i-1].Vx*W[i-1].d;
    qx = SIGN(qc)*MIN(2.0*MIN(fabs(qa),fabs(qb)), 0.5*fabs(qc));

    if ((-W[i].Vx*dWm[i][0]) > 0.0) {
      qa = 0.0;
      qb = -W[i].Vx*dWm[i][0];
    } else {
      qa = -W[i].Vx*dWm[i][0];
      qb = 0.0;
    }
    if (qx > 0.0) {
      qb += qx;
    } else {
      qa += qx;
    }
    qa = qa/W[i].d;
    qb = qb/W[i].d;

    dWm[i][1] = MIN(dWm[i][1],qb);
    dWm[i][1] = MAX(dWm[i][1],qa);
*/
  }
/*====================== END LOOP OVER il-2:il-1 =========================*/


/*--- Step 7. ------------------------------------------------------------------
 * Construct parabolic interpolant in primitive variables at left-interface
 * of cell il-1 ("W[il-3/2]", CW eqn 1.6) using linear TVD slopes at il-2 and
 * il-1 computed in Steps 2-7.
 */

  for (n=0; n<(NWAVE+NSCALARS); n++) {
    Wim1h[il-1][n] =.5*(pW[il-1][n]+pW[il-2][n])-(dWm[il-1][n]-dWm[il-2][n])/6.;
  }

/* Loop over il-2:il-1 in Steps 2-7 was necessary to bootstrap method by
 * computing Wim1h[il-1].  Now repeat these steps for rest of grid.  Splitting
 * into two loops avoids calculating eigensystems twice per cell.
 *
 * At the start of the loop, rem and lem still store values at i=il-1 computed
 * at the end of Step 2.  For each i, the eigensystem at i+1 is stored in
 * rem_ip1 and lem_ip1.  At the end of the loop rem[lem] is then set to
 * rem_ip1[lem_ip1] in preparation for the next iteration.
 */

/*========================= START BIG LOOP OVER i =========================*/
/* Steps 8-14 below are identical to steps 1-7 above */
  for (i=il-1; i<=iu+1; i++) {

/*--- Step 8. ------------------------------------------------------------------
 * Compute eigensystem in primitive variables.  */

#ifdef HYDRO
#ifdef ISOTHERMAL
    esys_prim_iso_hyd(W[i+1].d,W[i+1].Vx,         ev_ip1,rem_ip1,lem_ip1);
#else
    esys_prim_adb_hyd(W[i+1].d,W[i+1].Vx,W[i+1].P,ev_ip1,rem_ip1,lem_ip1);
#endif /* ISOTHERMAL */
#endif /* HYDRO */

#ifdef MHD
#ifdef ISOTHERMAL
    esys_prim_iso_mhd(W[i+1].d,W[i+1].Vx,
		       Bxc[i+1],W[i+1].By,W[i+1].Bz,ev_ip1,rem_ip1,lem_ip1);
#else
    esys_prim_adb_mhd(W[i+1].d,W[i+1].Vx,W[i+1].P,
		       Bxc[i+1],W[i+1].By,W[i+1].Bz,ev_ip1,rem_ip1,lem_ip1);
#endif /* ISOTHERMAL */
#endif /* MHD */

/*--- Step 9. ------------------------------------------------------------------
 * Compute centered, L/R, and van Leer differences of primitive variables */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      dWc[n] = pW[i+2][n] - pW[i][n];
      dWl[n] = pW[i+1][n] - pW[i][n];
      dWr[n] = pW[i+2][n] - pW[i+1][n];
      if (dWl[n]*dWr[n] > 0.0) {
        dWg[n] = 2.0*dWl[n]*dWr[n]/(dWl[n]+dWr[n]);
      } else {
        dWg[n] = 0.0;
      }
    }

/*--- Step 10. -----------------------------------------------------------------
 * Project differences in primitive variables along characteristics */

    for (n=0; n<NWAVE; n++) {
      dac[n] = lem_ip1[n][0]*dWc[0];
      dal[n] = lem_ip1[n][0]*dWl[0];
      dar[n] = lem_ip1[n][0]*dWr[0];
      dag[n] = lem_ip1[n][0]*dWg[0];
      for (m=1; m<NWAVE; m++) {
        dac[n] += lem_ip1[n][m]*dWc[m];
        dal[n] += lem_ip1[n][m]*dWl[m];
        dar[n] += lem_ip1[n][m]*dWr[m];
        dag[n] += lem_ip1[n][m]*dWg[m];
      }
    }

/* Advected variables are treated differently; for them the right and left
 * eigenmatrices are simply the identitiy matrix.
 */
#if (NSCALARS > 0)
    for (n=NWAVE; n<(NWAVE+NSCALARS); n++) {
      dac[n] = dWc[n];
      dal[n] = dWl[n];
      dar[n] = dWr[n];
      dag[n] = dWg[n];
    }
#endif

/*--- Step 11. -----------------------------------------------------------------
 * Apply monotonicity constraints to characteristic projections */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      da[n] = 0.0;
      if (dal[n]*dar[n] > 0.0) {
        lim_slope1 = MIN(    fabs(dal[n]),fabs(dar[n]));
        lim_slope2 = MIN(0.5*fabs(dac[n]),fabs(dag[n]));
        da[n] = SIGN(dac[n])*MIN(2.0*lim_slope1,lim_slope2);
      }
    }

/*--- Step 12. -----------------------------------------------------------------
 * Project monotonic slopes in characteristic back to primitive variables  */

    for (n=0; n<NWAVE; n++) {
      dWm[i+1][n] = da[0]*rem_ip1[n][0];
      for (m=1; m<NWAVE; m++) {
        dWm[i+1][n] += da[m]*rem_ip1[n][m];
      }
    }

#if (NSCALARS > 0)
    for (n=NWAVE; n<(NWAVE+NSCALARS); n++) {
      dWm[i+1][n] = da[n];
    }
#endif

/*--- Step 13. -----------------------------------------------------------------
 * When H-correction defined, limit velocity difference to sound speed
 * Limit velocity so momentum is always TVD (using only minmod limiter)
 * CURRENTLY NOT USED.  Was added to make code more robust for turbulence
 * simulations, but found it added noise to Noh shocktube.
 */

#ifdef H_CORRECTION
/* 
#ifdef ISOTHERMAL
    qa = Iso_csound;
#else
    qa = sqrt(Gamma*W[i+1].P/W[i+1].d);
#endif
    dWm[i+1][1] = SIGN(dWm[i+1][1])*MIN(fabs(dWm[i+1][1]),qa);
*/
#endif /* H_CORRECTION */
/* 
    qa = W[i+1].Vx*W[i+1].d - W[i  ].Vx*W[i  ].d;
    qb = W[i+2].Vx*W[i+2].d - W[i+1].Vx*W[i+1].d;
    qc = W[i+2].Vx*W[i+2].d - W[i  ].Vx*W[i  ].d;
    qx = SIGN(qc)*MIN(2.0*MIN(fabs(qa),fabs(qb)), 0.5*fabs(qc));

    if ((-W[i+1].Vx*dWm[i+1][0]) > 0.0) {
      qa = 0.0;
      qb = -W[i+1].Vx*dWm[i+1][0];
    } else {
      qa = -W[i+1].Vx*dWm[i+1][0];
      qb = 0.0;
    }
    if (qx > 0.0) {
      qb += qx;
    } else {
      qa += qx;
    }
    qa = qa/W[i+1].d;
    qb = qb/W[i+1].d;

    dWm[i+1][1] = MIN(dWm[i+1][1],qb);
    dWm[i+1][1] = MAX(dWm[i+1][1],qa);
*/
/*--- Step 14. -----------------------------------------------------------------
 * Construct parabolic interpolant in primitive variables at left-interface
 * of cell i+1 ("W[i+1/2]", CW eqn 1.6) using linear TVD slopes at i and
 * i+1 computed in Steps 2-7.
 */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      Wim1h[i+1][n] = 0.5*(pW[i+1][n]+pW[i][n]) - (dWm[i+1][n]-dWm[i][n])/6.0;
    }

/*--- Step 15. -----------------------------------------------------------------
 * Compute L/R values */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      Wlv[n] = Wim1h[i  ][n];
      Wrv[n] = Wim1h[i+1][n];
    }

/*--- Step 16. -----------------------------------------------------------------
 * Monotonize again (CW eqn 1.10), ensure they lie between neighboring
 * cell-centered vals */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      qa = (Wrv[n]-pW[i][n])*(pW[i][n]-Wlv[n]);
      qb = Wrv[n]-Wlv[n];
      qc = 6.0*(pW[i][n] - 0.5*(Wlv[n]+Wrv[n]));
      if (qa <= 0.0) {
        Wlv[n] = pW[i][n];
        Wrv[n] = pW[i][n];
      } else if ((qb*qc) > (qb*qb)) {
        Wlv[n] = 3.0*pW[i][n] - 2.0*Wrv[n];
      } else if ((qb*qc) < -(qb*qb)) {
        Wrv[n] = 3.0*pW[i][n] - 2.0*Wlv[n];
      }
    }

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      Wlv[n] = MAX(MIN(pW[i][n],pW[i-1][n]),Wlv[n]);
      Wlv[n] = MIN(MAX(pW[i][n],pW[i-1][n]),Wlv[n]);
      Wrv[n] = MAX(MIN(pW[i][n],pW[i+1][n]),Wrv[n]);
      Wrv[n] = MIN(MAX(pW[i][n],pW[i+1][n]),Wrv[n]);
    }

/*--- Step 17. -----------------------------------------------------------------
 * Compute coefficients of interpolation parabolae (CW eqn 1.5) */

    for (n=0; n<(NWAVE+NSCALARS); n++) {
      dW[n] = Wrv[n] - Wlv[n];
      W6[n] = 6.0*(pW[i][n] - 0.5*(Wlv[n] + Wrv[n]));
    }

/*--- Step 18. -----------------------------------------------------------------
 * Integrate linear interpolation function over domain of dependence defined by
 * max(min) eigenvalue (CW eqn 1.12)
 */

    pWl = (Real *) &(Wl[i+1]);
    pWr = (Real *) &(Wr[i]);

    qx = TWO_3RDS*MAX(ev[NWAVE-1],0.0)*dtodx;
    for (n=0; n<(NWAVE+NSCALARS); n++) {
      pWl[n] = Wrv[n] - 0.75*qx*(dW[n] - (1.0 - qx)*W6[n]);
    }

    qx = -TWO_3RDS*MIN(ev[0],0.0)*dtodx;
    for (n=0; n<(NWAVE+NSCALARS); n++) {
      pWr[n] = Wlv[n] + 0.75*qx*(dW[n] + (1.0 - qx)*W6[n]);
    }

#ifndef THREED_VL /* include step 20 only if not using VL 3D integrator */
/* NB: If THREED_VL is defined, cannot run 1D or 2D problems; see integrate.c */

/*--- Step 19. -----------------------------------------------------------------
 * Then subtract amount of each wave m that does not reach the interface
 * during timestep (CW eqn 3.5ff).  For HLL fluxes, must subtract waves that
 * move in both directions, but only to 2nd order.
 */

    for (n=0; n<NWAVE; n++) {
      if (ev[n] > 0.) {
	qa  = 0.0;
        qb = 0.5*dtodx*(ev[NWAVE-1]-ev[n]);
        qc = 0.5*dtodx*dtodx*TWO_3RDS*(ev[NWAVE-1]*ev[NWAVE-1] - ev[n]*ev[n]);
	for (m=0; m<NWAVE; m++) {
	  qa += lem[n][m]*(qb*(dW[m]-W6[m]) + qc*W6[m]);
	}
	for (m=0; m<NWAVE; m++) pWl[m] += qa*rem[m][n];
/* For HLL fluxes, subtract wave moving away from interface as well. */
#if defined(HLLE_FLUX) || defined(HLLC_FLUX) || defined(HLLD_FLUX)
        qa  = 0.0;
        qb = 0.5*dtodx*(ev[0]-ev[n]);
        qc = 0.5*dtodx*dtodx*TWO_3RDS*(ev[0]*ev[0] - ev[n]*ev[n]);
        for (m=0; m<NWAVE; m++) {
          qa += lem[n][m]*(qb*(dW[m]+W6[m]) + qc*W6[m]);
        }

        for (m=0; m<NWAVE; m++) pWr[m] += qa*rem[m][n];
#endif /* HLL_FLUX */
      }
    }

    for (n=0; n<NWAVE; n++) {
      if (ev[n] < 0.) {
        qa = 0.0;
        qb = 0.5*dtodx*(ev[0]-ev[n]);
        qc = 0.5*dtodx*dtodx*TWO_3RDS*(ev[0]*ev[0] - ev[n]*ev[n]);
        for (m=0; m<NWAVE; m++) {
          qa += lem[n][m]*(qb*(dW[m]+W6[m]) + qc*W6[m]);
        }
        for (m=0; m<NWAVE; m++) pWr[m] += qa*rem[m][n];
/* For HLL fluxes, subtract wave moving away from interface as well. */
#if defined(HLLE_FLUX) || defined(HLLC_FLUX) || defined(HLLD_FLUX)
        qa = 0.0;
        qb = 0.5*dtodx*(ev[NWAVE-1]-ev[n]);
        qc = 0.5*dtodx*dtodx*TWO_3RDS*(ev[NWAVE-1]*ev[NWAVE-1] - ev[n]*ev[n]);
        for (m=0; m<NWAVE; m++) {
          qa += lem[n][m]*(qb*(dW[m]-W6[m]) + qc*W6[m]);
        }

        for (m=0; m<NWAVE; m++) pWl[m] += qa*rem[m][n];
#endif /* HLL_FLUX */
      }
    }

/* Wave subtraction for advected quantities */
    for (n=NWAVE; n<(NWAVE+NSCALARS); n++) {
      if (W[i].Vx > 0.) {
        qb = 0.5*dtodx*(ev[NWAVE-1]-W[i].Vx);
        qc = 0.5*dtodx*dtodx*TWO_3RDS*(SQR(ev[NWAVE-1]) - SQR(W[i].Vx));
        pWl[n] += qb*(dW[m]-W6[m]) + qc*W6[m];
      } else if (W[i].Vx < 0.) {
        qb = 0.5*dtodx*(ev[0]-W[i].Vx);
        qc = 0.5*dtodx*dtodx*TWO_3RDS*(ev[0]*ev[0] - W[i].Vx*W[i].Vx);
        pWr[n] += qb*(dW[m]+W6[m]) + qc*W6[m];
      }
    }

#endif /* THREED_VL */

/*--- Step 20. -----------------------------------------------------------------
 * Save eigenvalues and eigenmatrices at i+1 for use in next iteration */

    for (m=0; m<NWAVE; m++) {
      ev[m] = ev_ip1[m];
      for (n=0; n<NWAVE; n++) {
        rem[m][n] = rem_ip1[m][n];
        lem[m][n] = lem_ip1[m][n];
      }
    }

  } /*====================== END BIG LOOP OVER i =========================*/

  return;
}

/*----------------------------------------------------------------------------*/
/* lr_states_init:  Allocate enough memory for work arrays */

void lr_states_init(int nx1, int nx2, int nx3)
{
  int nmax;
  nmax =  nx1 > nx2  ? nx1 : nx2;
  nmax = (nx3 > nmax ? nx3 : nmax) + 2*nghost;

  if ((pW = (Real**)malloc(nmax*sizeof(Real*))) == NULL) goto on_error;

  if ((dWm = (Real**)calloc_2d_array(nmax, NWAVE, sizeof(Real))) == NULL)
    goto on_error;

  if ((Wim1h = (Real**)calloc_2d_array(nmax, NWAVE, sizeof(Real))) == NULL)
    goto on_error;

  return;
  on_error:
    lr_states_destruct();
    ath_error("[lr_states_init]: malloc returned a NULL pointer\n");
}

/*----------------------------------------------------------------------------*/
/* lr_states_destruct:  Free memory used by work arrays */

void lr_states_destruct(void)
{
  if (pW != NULL) free(pW);
  if (dWm != NULL) free_2d_array(dWm);
  if (Wim1h != NULL) free_2d_array(Wim1h);
  return;
}

#endif /* THIRD_ORDER */