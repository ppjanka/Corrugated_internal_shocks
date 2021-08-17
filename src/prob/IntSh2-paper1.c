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
static inline Real v2gamma (Real vel)
{return 1./sqrt(1.-vel*vel);}
static inline Real gamma2v (Real _gamma)
{return sqrt(1.-1./(_gamma*_gamma));}

static inline Real sigmoid (Real x)
{return 1./(1.+exp(-x));}

/*----------------------------------------------------------------------------*/
/* problem:  */

void problem(DomainS *pDomain)
{

  GridS *pGrid=(pDomain->Grid);
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  int k, ks = pGrid->ks, ke = pGrid->ke;
  Real z,r,x3;

  // read out or default the floors
  dfloor = par_getd_def("problem", "dfloor", TINY_NUMBER);
  pfloor = par_getd_def("problem", "pfloor", TINY_NUMBER);

  // read out the initial conditions
	Real adiab_idx = par_getd("problem", "gamma");
  // ambient medium
	Real rho_amb = par_getd("problem", "rho_amb");
  Real press_amb = par_getd("problem", "press_amb");
  Real bfield_amb = par_getd("problem", "bfield_amb");
  // shells
  Real xcen_sh [2] =
          {par_getd("problem", "xcen_sh1"), par_getd("problem", "xcen_sh2")};
  Real width_sh [2] =
          {par_getd("problem", "width_sh1"), par_getd("problem", "width_sh2")};
  Real x1_sh [2], x2_sh [2];
  Real vel_sh [2] =
          {par_getd("problem", "vel_sh1"), par_getd("problem", "vel_sh2")};
  Real gamma_sh [2];
  Real rho_sh [2] =
          {par_getd("problem", "rho_sh1"), par_getd("problem", "rho_sh2")};
  Real press_sh [2] =
          {par_getd("problem", "press_sh1"), par_getd("problem", "press_sh2")};
  #ifdef MHD
  Real sigmaB_sh [2] =
          {par_getd("problem", "sigmaB_sh1"), par_getd("problem", "sigmaB_sh2")};
  Real bfield_sh [2];
  Real B, sqr_b;
  #endif
  // auxiliary variables
  #pragma omp simd
  for (i=0; i<2; i++) {
    x1_sh[i] = xcen_sh[i] - 0.5*width_sh[i];
    x2_sh[i] = xcen_sh[i] + 0.5*width_sh[i];
    gamma_sh[i] = v2gamma(vel_sh[i]);
    #ifdef MHD
    bfield_sh[i] = sqrt(2.*rho_sh[i]*sigmaB_sh[i]) * gamma_sh[i];
    #endif
  }
  // corrugation
  Real corr_ampl = par_getd_def("problem", "corr_ampl", 0.0);
  int corr_nx = par_geti_def("problem", "corr_nx", 2);
  int corr_ny = par_geti_def("problem", "corr_ny", 2);

  // set the initial conditions
  Real rho, press, vel, gamma, sqr_gamma, enthalpy;
	for (k = ks; k <= ke; k++) {
    for (j = js; j <= je+1; j++) {
      #pragma omp simd
      for (i = is; i <= ie+1; i++) {

        // read the current location
        cc_pos(pGrid,i,j,k,&z,&r,&x3);

        if (z < x1_sh[0]) { // left of first shell
          rho = rho_amb; press = press_amb; vel = vel_sh[0]; gamma = gamma_sh[0];
          #ifdef MHD
          B = 0.0;
          #endif
          if (press < 0) { // ensure pressure equilibrium
            press = press_sh[0]
              #ifdef MHD
                + 0.5 * (SQR(bfield_sh[0]) - SQR(B)) / SQR(gamma);
              #else
                ;
              #endif
          }
        } else if (z <= x2_sh[0]) { // inside first shell
          rho = rho_sh[0]; press = press_sh[0]; vel = vel_sh[0]; gamma = gamma_sh[0];
          #ifdef MHD
          B = bfield_sh[0];
          #endif
        } else if (z < x1_sh[1]) { // between shells
          rho = rho_amb; press = press_amb;
          #ifdef MHD
          B = 0.0;
          #endif
          /*vel = sigmoid(z*(2*M_PI/(x1_sh[1]-x2_sh[0])))
                      * (vel_sh[1]-vel_sh[0]) + vel_sh[0];
          gamma = v2gamma(vel);*/
          vel = 0.0;
          gamma = 1.0;
          if (press < 0) { // ensure pressure equilibrium
            press = sigmoid(z*(2*M_PI/(x1_sh[1]-x2_sh[0])))
                          * (press_sh[1]-press_sh[0]) + press_sh[0]
            #ifdef MHD
              + 0.5 * (SQR(
                            sigmoid(z*(2*M_PI/(x1_sh[1]-x2_sh[0])))
                              * (bfield_sh[1]-bfield_sh[0]) + bfield_sh[0]
                         )
                  - SQR(B)) / SQR(gamma);
            #else
              ;
            #endif
          }
          if (corr_ampl > 0.0) { // apply corrugation
            rho += (corr_ampl - rho_amb)
                * 0.5 * ( cos( 2.*M_PI *
                     (corr_nx * (z-x2_sh[0]) / (x1_sh[1]-x2_sh[0])
                    + corr_ny * (r-pDomain->MinX[1]) / (pDomain->MaxX[1]-pDomain->MinX[1]))
                 - M_PI) +1 );
          }
        } else if (z <= x2_sh[1]) { // inside second shell
          rho = rho_sh[1]; press = press_sh[1]; vel = vel_sh[1]; gamma = gamma_sh[1];
          #ifdef MHD
          B = bfield_sh[1];
          #endif
        } else { // right of second shell
          rho = rho_amb; press = press_amb; vel = vel_sh[1]; gamma = gamma_sh[1];
          #ifdef MHD
          B = 0.0;
          #endif
          if (press < 0) { // ensure pressure equilibrium
            press = press_sh[1]
              #ifdef MHD
                + 0.5 * (SQR(bfield_sh[0]) - SQR(B)) / SQR(gamma);
              #else
                ;
              #endif
          }
        }

        // Set the hydro parameters
        // NOTE: rho and Pg are in the fluid frame, Beckwith & Stone 2011
        sqr_gamma = SQR(gamma);
        enthalpy = 1. + adiab_idx * press / ((adiab_idx-1.)*rho);
        pGrid->U[k][j][i].d = gamma*rho;
        pGrid->U[k][j][i].M1 = sqr_gamma*rho*enthalpy*vel;
        pGrid->U[k][j][i].M2 = 0.;
        pGrid->U[k][j][i].M3 = 0.;
        pGrid->U[k][j][i].E = sqr_gamma*rho*enthalpy - press;

        #ifdef MHD
        // set bfield in the y-direction
        pGrid->U[k][j][i].B1c = 0.;
        pGrid->U[k][j][i].B2c = B;
        pGrid->U[k][j][i].B3c = 0.;
        if (i >= is) {
          pGrid->B2i[k][j][i] = B;
        }

        // make adjustments due to bfield
        sqr_b = SQR(pGrid->U[k][j][i].B1c) +
                 SQR(pGrid->U[k][j][i].B2c) +
                 SQR(pGrid->U[k][j][i].B3c);
        pGrid->U[k][j][i].M1 += sqr_b * vel /*from w_tot*/;
        pGrid->U[k][j][i].E += sqr_b /*from w_tot*/ - 0.5 * sqr_b / sqr_gamma /*from P_tot*/;
        #endif //MHD
      }
    }
  }

  #ifdef MHD
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
  #endif MHD

  //exit(0);
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
