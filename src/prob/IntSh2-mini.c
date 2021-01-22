#include "copyright.h"
/*============================================================================*/
/*! \file current_sheet.c
 *  \brief Problem generator for current sheet test. 
 *
 * PURPOSE: Problem generator for current sheet test.  This version only allows
 *   current sheet in X-Y plane, with Bz=0.  
 *
 * REFERENCE: */
/*============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

// Special Relativity handling functions
inline Real v2gamma (Real vel)
{return 1./sqrt(1.-vel*vel);}
inline Real gamma2v (Real _gamma)
{return sqrt(1.-1./(_gamma*_gamma));}

// declarations
void inflow_boundary (GridS *pGrid);

/*----------------------------------------------------------------------------*/
/* problem:  */

void problem(DomainS *pDomain)
{

  GridS *pGrid=(pDomain->Grid);
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  int k, ks = pGrid->ks, ke = pGrid->ke;
  Real z,r,x3;

  // read out the initial conditions
	Real rmax = par_getd("domain1", "x2max");
	Real adiab_idx = par_getd("problem", "gamma");
  // pre-shock conditions
  Real rho1 = par_getd("problem", "rho1");
  Real press1 = par_getd("problem", "press1");
  Real gamma1 = par_getd("problem", "gamma1");
  Real bfield1 = par_getd("problem", "bfield1");
  // post-shock conditions
  Real rho2 = par_getd("problem", "rho2");
  Real press2 = par_getd("problem", "press2");
  Real gamma2 = par_getd("problem", "gamma2");
  Real bfield2 = par_getd("problem", "bfield2");
	// shock position
	Real z_shock = par_getd("problem", "z_shock");

  Real rho, press, vel, B, _gamma, sqr_gamma, enthalpy;
  Real sqr_b;

	for (k = ks; k <= ke; k++) {
    for (j = js; j <= je+1; j++) {
      for (i = ie; i >= is-1; i--) {

        // read the current location
        cc_pos(pGrid,i,j,k,&z,&r,&x3);

        if (z > z_shock) { // pre-shock
          rho = rho1; press = press1; _gamma = gamma1; B = bfield1;
        } else { // post-shock
          rho = rho2; press = press2; _gamma = gamma2; B = bfield2;
        }

        // Set the hydro parameters
        // NOTE: rho and Pg are in the fluid frame, Beckwith & Stone 2011
        vel = - gamma2v(_gamma);
        enthalpy = 1. + adiab_idx * press / ((adiab_idx-1.)*rho);
        pGrid->U[k][j][i].d = _gamma*rho;
        pGrid->U[k][j][i].M1 = _gamma*_gamma*rho*enthalpy*vel;
        pGrid->U[k][j][i].M2 = 0.;
        pGrid->U[k][j][i].M3 = 0.;
        pGrid->U[k][j][i].E = _gamma*_gamma*rho*enthalpy - press;

        // set bfield in the y-direction
        pGrid->U[k][j][i].B1c = 0.;
        pGrid->U[k][j][i].B2c = B;
        pGrid->U[k][j][i].B3c = 0.;
        if (i >= is) {
          pGrid->B2i[k][j][i] = B;
        }

        // make adjustments due to bfield
        sqr_gamma = SQR(_gamma);
        sqr_b = SQR(pGrid->U[k][j][i].B1c) +
                 SQR(pGrid->U[k][j][i].B2c) +
                 SQR(pGrid->U[k][j][i].B3c);
        pGrid->U[k][j][i].M1 += sqr_b * vel /*from w_tot*/;
        pGrid->U[k][j][i].E += sqr_b /*from w_tot*/ - 0.5 * sqr_b / sqr_gamma /*from P_tot*/;

      }
    }
  }

	// set the rest of bfield
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie+1; i++) {
        pGrid->B1i[k][j][i] = 0.0;
      }
    }
  }

  for (k=ks; k<=(ke > 1 ? ke+1 : ke); k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {
        pGrid->B3i[k][j][i] = 0.0;
      }
    }
  }

  // enroll the bvals function
  bvals_mhd_fun(pDomain, right_x1, inflow_boundary);

}

/*==============================================================================
 * PROBLEM USER FUNCTIONS:
 * problem_write_restart() - writes problem-specific user data to restart files
 * problem_read_restart()  - reads problem-specific user data from restart files
 * get_usr_expr()          - sets pointer to expression for special output data
 * get_usr_out_fun()       - returns a user defined output function pointer
 * get_usr_par_prop()      - returns a user defined particle selection function
 * Userwork_in_loop        - problem specific work IN     main loop
 * Userwork_after_loop     - problem specific work AFTER  main loop
 *----------------------------------------------------------------------------*/

void problem_write_restart(MeshS *pM, FILE *fp)
{
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  // enroll the bvals function
  bvals_mhd_fun(*(pM->Domain), right_x1, inflow_boundary);

  return;
}

ConsFun_t get_usr_expr(const char *expr)
{
  return NULL;
}

VOutFun_t get_usr_out_fun(const char *name){
  return NULL;
}

void Userwork_in_loop(MeshS *pM)
{
}

void Userwork_after_loop(MeshS *pM)
{
}

void inflow_boundary (GridS *pGrid) {

  int i, il = pGrid->is, iu = pGrid->ie;
  int j, jl = pGrid->js, ju = pGrid->je;
  int k, kl = pGrid->ks, ku = pGrid->ke;

  Real time = pGrid->time;

  Real z,r,x3;

  // read out the initial conditions
  Real rmax = par_getd("domain1", "x2max");
  Real adiab_idx = par_getd("problem", "gamma");
  // pre-shock conditions
  Real rho1 = par_getd("problem", "rho1");
  Real press1 = par_getd("problem", "press1");
  Real gamma1 = par_getd("problem", "gamma1");
  Real bfield1 = par_getd("problem", "bfield1");
  // shock position
  Real z_shock = par_getd("problem", "z_shock");
  // shock corrugation parameters: cos terms
  Real corr_cos1_A = par_getd("problem", "corr_cos1_A");
  Real corr_cos1_L = par_getd("problem", "corr_cos1_L");
  Real corr_cos1_f = 2.*M_PI / corr_cos1_L;
  Real corr_cos1_Lz = par_getd("problem", "corr_cos1_Lz");
  if (corr_cos1_Lz == 0) corr_cos1_Lz = HUGE_NUMBER;
  Real corr_cos1_fz = 2.*M_PI / corr_cos1_Lz;

  // initialize the grid conditions
  Real rho, press, vel, _gamma, B;
  press = press1; _gamma = gamma1, B = bfield1;
  vel = - gamma2v(_gamma);

  Real corr, enthalpy, sqr_gamma, sqr_b;

  for (k = kl; k <= ku; k++) {
    for (j = jl; j <= ju; j++) {
      //#pragma omp simd
      for (i = iu; i <= iu+nghost; i++) {

        // read the current location
        cc_pos(pGrid,i,j,k,&z,&r,&x3);

        // calculate density corrugation
        corr = corr_cos1_A * cos(corr_cos1_f * r + corr_cos1_fz * (z-z_shock-vel*time));
        rho = rho1 * (1.0 + corr);

        // Set the hydro parameters
        // NOTE: rho and Pg are in the fluid frame, Beckwith & Stone 2011
        enthalpy = 1. + adiab_idx * press / ((adiab_idx-1.)*rho);
        pGrid->U[k][j][i].d = _gamma*rho;
        pGrid->U[k][j][i].M1 = _gamma*_gamma*rho*enthalpy*vel;
        pGrid->U[k][j][i].M2 = 0.;
        pGrid->U[k][j][i].M3 = 0.;
        pGrid->U[k][j][i].E = _gamma*_gamma*rho*enthalpy - press;

        // set bfield in the y-direction
        pGrid->U[k][j][i].B1c = 0.;
        pGrid->U[k][j][i].B2c = B;
        pGrid->U[k][j][i].B3c = 0.;

        // make adjustments due to bfield
        sqr_gamma = SQR(_gamma);
        sqr_b = SQR(pGrid->U[k][j][i].B1c) +
                 SQR(pGrid->U[k][j][i].B2c) +
                 SQR(pGrid->U[k][j][i].B3c);
        pGrid->U[k][j][i].M1 += sqr_b * vel /*from w_tot*/;
        pGrid->U[k][j][i].E += sqr_b /*from w_tot*/ - 0.5 * sqr_b / sqr_gamma /*from P_tot*/;

      }
    }
  }

  // initialize constant By
  for (k = kl; k <= ku; k++) {
    for (j = jl; j <= ju; j++) {
      #pragma omp simd
      for (i = iu+1; i <= iu+nghost+1; i++) {
        pGrid->B1i[k][j][i] = 0.0;
      }
    }
  }
  for (k = kl; k <= ku; k++) {
    for (j = jl; j <= ju+1; j++) {
      #pragma omp simd
      for (i = iu; i <= iu+nghost; i++) {
        pGrid->B2i[k][j][i] = B;
      }
    }
  }
  for (k = kl; k <= (ku > 1 ? ku+1 : ku); k++) {
    for (j = jl; j <= ju; j++) {
      #pragma omp simd
      for (i = iu; i <= iu+nghost; i++) {
        pGrid->B3i[k][j][i] = 0.0;
      }
    }
  }

}
