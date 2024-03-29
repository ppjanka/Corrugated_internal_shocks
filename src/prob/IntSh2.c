#include "copyright.h"
/*============================================================================*/
/*! \file IntSh2.c
 *  \brief Problem generator for the corrugated internal shock collision simulations
 *  in context of relativistic jets in microquasars.
 *  Author: Patryk Pjanka, Nordita, 2021 */
/*============================================================================*/

//#define DEBUG

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"

#include <gsl/gsl_poly.h> // NOTE: -lgsl -lgslcblas -lm needed at linking stage

// Special Relativity handling functions
inline Real v2gamma (Real vel)
{return 1./sqrt(1.-vel*vel);}
inline Real gamma2v (Real _gamma)
{return sqrt(1.-1./(_gamma*_gamma));}

// declarations
void inflow_boundary (GridS *pGrid);

// Solver for the post-shock conditions from relativistic Rankine-Hugoniot conditions
// -- post-shock Lorentz factor space, with Bfield included
void postshock_gamma_bfield (Real adiab_idx,
		Real rho1, Real press1, Real vel1, Real gamma1, Real B1,
		Real* rho2, Real* press2, Real* vel2, Real* gamma2, Real* B2) {

  #ifdef DEBUG
  printf("gamma1 = %.2e\n", gamma1);
  #endif

  // calculate pre-shock constants
  Real D = B1 * vel1; // magnetic flux
  Real A = gamma1 * rho1 * vel1; // effective mass flux
  Real B = (rho1 + adiab_idx * press1 / (adiab_idx - 1.0)) * SQR(gamma1*vel1)
		  + press1 + SQR(D) * (1.0 + 0.5 / SQR(vel1*gamma1) ); // momentum flux
  Real C = (rho1 + adiab_idx * press1 / (adiab_idx - 1.0)) * SQR(gamma1) * vel1
		  + SQR(D) / vel1; // energy flux

  #ifdef DEBUG
  printf("vel1 = %.2e, A = %.2e, B = %.2e, C = %.2e, D = %.2e\n", vel1, A, B, C, D);
  #endif

  // calculate root equation's factors
  Real a0, a1, a2, a3, a4, a5, a6;
  a6 = SQR(adiab_idx/(adiab_idx-1.0)) * (SQR(C)-SQR(B));
  a5 = - 2.0*adiab_idx * A*C /(adiab_idx-1.0);
  a4 = SQR(A) - adiab_idx*(adiab_idx+2.0) * SQR(C/(adiab_idx-1.0)) + 2.0 * SQR(adiab_idx * B /(adiab_idx-1.0))
        - (2.0*adiab_idx/(adiab_idx-1.0)) * (1.0 - 0.5*adiab_idx/(adiab_idx-1.0)) * B*SQR(D);
  a3 = 2.0 * (adiab_idx+1.0) * A*C / (adiab_idx-1.0);
  a2 = -SQR(A) + (2.0*adiab_idx+1.0) * SQR(C/(adiab_idx-1.0)) - SQR(adiab_idx*B/(adiab_idx-1.0))
        + (2.0*adiab_idx/(adiab_idx-1.0)) * (1.0 - 0.5*adiab_idx/(adiab_idx-1.0)) * B*SQR(D)
        - SQR(1.0 - 0.5*adiab_idx/(adiab_idx-1.0)) * pow(D,4);
  a1 = -2.0 * A*C / (adiab_idx - 1.0);
  a0 = - SQR(C/(adiab_idx-1.0));

  #ifdef DEBUG
  printf("a6 = %.2e, a5 = %.2e, a4 = %.2e, a3 = %.2e, a2 = %.2e, a1 = %.2e, a0 = %.2e\n", a6, a5, a4, a3, a2, a1, a0);
  #endif

  // solve the quartic equation for vel2
  Real a[7] = {a0,a1,a2,a3,a4,a5,a6};
  Real z[12]; // solutions
  gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc (7);
  gsl_poly_complex_solve (a, 7, w, z);
  gsl_poly_complex_workspace_free (w);

  Real gamma2_here, vel2_here, rho2_here, press2_here, B2_here;

  #ifdef DEBUG
  printf("All solutions for gamma (real,complex) pairs:\n");
  for (int i = 0; i < 6; i++) {
    printf("  %.2e, %.2e, equation(Re(z)) = %.2e\n", z[2*i], z[2*i+1]
             , a0 + a1*z[2*i] + a2*SQR(z[2*i]) + a3*pow(z[2*i],3) + a4*pow(z[2*i],4) + a5*pow(z[2*i],5) + a6*pow(z[2*i],6));

    gamma2_here = z[2*i];

    // calculate the remaining postshock conditions
    (*vel2) = - gamma2v(gamma2_here);
    (*rho2) = A / ((*vel2)*gamma2_here);
    (*B2) = vel1*B1 / (*vel2);
    (*press2) = B - C*(*vel2) - 0.5*SQR((*B2)/gamma2_here);

    // check the solution
    Real D2 = (*B2) * (*vel2); // magnetic flux
    Real A2 = gamma2_here * (*rho2) * (*vel2); // effective mass flux
    Real B2x = ((*rho2) + adiab_idx * (*press2) / (adiab_idx - 1.0)) * SQR(gamma2_here*(*vel2)) + (*press2) + SQR(D2) * (1.0 + 0.5 / SQR((*vel2)*gamma2_here) ); // momentum flux
    Real C2 = ((*rho2) + adiab_idx * (*press2) / (adiab_idx - 1.0)) * SQR(gamma2_here) * (*vel2) + SQR(D2) / (*vel2); // energy flux

    printf("    A = %.2e = %.2e\n", A, A2);
    printf("    B = %.2e = %.2e\n", B, B2x);
    printf("    C = %.2e = %.2e\n", C, C2);
    printf("    D = %.2e = %.2e\n", D, D2);
  }
  #endif

  // there will only be one physical solution, with (*vel2) in [vel1,0] (vel1 < 0), so keep trying until it is found
  int solution_counter = 0;
  for (int i = 0; i < 6; i++) {

    gamma2_here = z[2*i];

    if (fabs(z[2*i+1]) > 1.0e-6 || gamma2_here < 1.0) continue; // complex and unphysical solutions

    if (fabs(gamma2_here-gamma1) < 1.0e-6) continue; // contact discontinuity

    if (gamma2_here > gamma1) continue; // unphysical

    // calculate the remaining postshock conditions
    vel2_here = - gamma2v(gamma2_here);
    rho2_here = A / (vel2_here*gamma2_here);
    B2_here = vel1*B1 / vel2_here;
    press2_here = B - C*vel2_here - 0.5*SQR(B2_here/gamma2_here);

    // check the solution
    Real D2 = B2_here * vel2_here; // magnetic flux
    Real A2 = gamma2_here * rho2_here * vel2_here; // effective mass flux
    Real B2x = (rho2_here + adiab_idx * press2_here / (adiab_idx - 1.0)) * SQR(gamma2_here*vel2_here) + press2_here + SQR(D2) * (1.0 + 0.5 / SQR(vel2_here*gamma2_here) ); // momentum flux
    Real C2 = (rho2_here + adiab_idx * press2_here / (adiab_idx - 1.0)) * SQR(gamma2_here) * vel2_here + SQR(D2) / vel2_here; // energy flux

    if (fabs(A-A2) > 1.0e-5 || fabs(B-B2x) > 1.0e-5 || fabs(C-C2) > 1.0e-5 || fabs(D-D2) > 1.0e-5) {
      continue; // shock in the opposite direction (v > 0)
    } else {
      (*gamma2) = gamma2_here;
      (*vel2) = - gamma2v((*gamma2));
      (*rho2) = A / ((*vel2)*(*gamma2));
      (*B2) = vel1*B1 / (*vel2);
      (*press2) = B - C*(*vel2) - 0.5*SQR((*B2)/(*gamma2));
      solution_counter++;
    }

  }
  if (solution_counter == 0) {
    printf("[intsh2.cpp:postshock(...)] ERROR: No physical postshock solution found. Aborting.");
    exit(0);
  }
  if (solution_counter > 1) {
    printf("[intsh2.cpp:postshock(...)] ERROR: More than one viable postshock solution found. Aborting.");
    exit(0);
  }

  #ifdef DEBUG
  printf("gamma2 = %.2e, vel2 = %.2e, rho2 = %.2e, press2 = %.2e, B2 = %.2e\n", *gamma2, *vel2, *rho2, *press2, *B2);
  //exit(0);
  #endif

}

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
	Real r_jet = par_getd("problem", "r_jet");
	Real rmax = par_getd("domain1", "x2max");
	Real adiab_idx = par_getd("problem", "gamma");
	// ambient medium
	Real rho_amb = par_getd("problem", "rho_amb");
	Real press_amb = par_getd("problem", "press_amb");
	// central jet conditions
	Real rho_jet = par_getd("problem", "rho_jet");
	Real press_jet = par_getd("problem", "press_jet");
	// shock injection parameters
	Real gamma_shock1 = par_getd("problem", "gamma_shock1");
	Real z_shock1 = par_getd("problem", "z_shock1");
  Real gamma_shock2 = par_getd("problem", "gamma_shock2");
  Real z_shock2 = par_getd("problem", "z_shock2");
  Real gamma_shock2_LAB, vel_shock2_LAB;
	// magnetic field parameters for constant By
	Real bfield_A = par_getd("problem", "bfield_A");

  Real eta, enthalpy;
  Real rho, press, vel, B, _gamma;
  Real rho1, press1, vel1, B1, gamma1;
  Real sqr_b, sqr_gamma;

	for (k = ks; k <= ke; k++) {
    for (j = js; j <= je+1; j++) {
      for (i = ie; i >= is-1; i--) {

        // read the current location
        cc_pos(pGrid,i,j,k,&z,&r,&x3);

        // radial profile
        eta = 0.5 + 0.5*cos(3.*r/rmax);

        // Calculate fluid properties from large-x1 to small-x1 ("right-to-left")
        // -- i.e., in the direction of the flow

        // 0: Pre-shock--------------------------------------------------------
        rho = eta * (rho_jet - rho_amb) + rho_amb;
        press = eta * (press_jet - press_amb) + press_amb;
        _gamma = gamma_shock1;
        vel = - gamma2v(_gamma);
        B = bfield_A;

        // 1: Post-shock1------------------------------------------------------
        if (z < z_shock1) {
          rho1 = rho; press1 = press; vel1 = vel; gamma1 = _gamma; B1 = B;
          postshock_gamma_bfield(adiab_idx, rho1, press1, vel1, gamma1, B1,
                &rho, &press, &vel, &_gamma, &B);
        }
        // 1: Post-shock2-----------------------------------------------------
        if (z < z_shock2 && fabs(gamma_shock2) > 1.0e-6) {
          rho1 = rho; press1 = press; vel1 = vel; gamma1 = _gamma; B1 = B;
          // calculate the 2nd shock velocity in the LAB frame
          gamma_shock2_LAB = ((gamma_shock2/gamma1) - fabs(vel)*sqrt(SQR((gamma_shock2/gamma1)) + SQR(vel1) - 1.0)) / (1.0 - SQR(vel1));
          vel_shock2_LAB = gamma2v(gamma_shock2_LAB);
          // transform pre-shock Bfield to the new shock's frame
          B1 = B1 * gamma_shock2 / gamma1;
          // transform pre-shock velocity to the new shock's frame
          gamma1 = gamma_shock2_LAB * gamma1 * (1.0 - vel_shock2_LAB*vel1); // should equal gamma_shock2
          vel1 = - gamma2v(gamma1);
          #ifdef DEBUG
          printf("gsh2 = %.2e, g1 = %.2e, vsh2 = %.2e, v1 = %.2e\n", gamma_shock2_LAB, gamma1, vel_shock2_LAB, vel1);
          printf("gamma1 = %.2e = %.2e = gamma_shock2\n", gamma1, gamma_shock2);
          #endif
          // apply RH conditions
          postshock_gamma_bfield(adiab_idx, rho1, press1, vel1, gamma1, B1,
              &rho, &press, &vel, &_gamma, &B);
          // TODO: transform post-shock Bfield back to the LAB frame
          B *= gamma_shock2_LAB * (1.0 + vel_shock2_LAB*vel); // = gamma_LAB / gamma
          // transform post-shock velocity back to the LAB frame
          _gamma *= gamma_shock2_LAB * (1.0 + vel_shock2_LAB*vel); // = gamma_LAB / gamma
          vel = - gamma2v(_gamma);
          #ifdef DEBUG
          exit(0);
          #endif
        }

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

	// set the rest of face-centered bfield
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      #pragma omp simd
      for (i=is; i<=ie+1; i++) {
        pGrid->B1i[k][j][i] = 0.0;
      }
    }
  }

  for (k=ks; k<=(ke > 1 ? ke+1 : ke); k++) {
    for (j=js; j<=je; j++) {
      #pragma omp simd
      for (i=is; i<=ie; i++) {
        pGrid->B3i[k][j][i] = 0.0;
      }
    }
  }

  // enroll the bvals inflow function if we touch the inflow boundary
  fc_pos(pGrid,ie+1,0,0,&z,&r,&x3);
  if ((z - par_getd("domain1", "x1max")) < 1.0e-6) {
    bvals_mhd_fun(pDomain, right_x1, inflow_boundary);
  }

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
  if ((*pM->Domain)->Level == 0) {
    bvals_mhd_fun(*(pM->Domain), right_x1, inflow_boundary);
  }

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
  Real r_jet = par_getd("problem", "r_jet");
  Real rmax = par_getd("domain1", "x2max");
  Real adiab_idx = par_getd("problem", "gamma");
  // ambient medium
  Real rho_amb = par_getd("problem", "rho_amb");
  Real press_amb = par_getd("problem", "press_amb");
  // central jet conditions
  Real rho_jet = par_getd("problem", "rho_jet");
  Real press_jet = par_getd("problem", "press_jet");
  // shock injection parameters
  Real gamma_shock1 = par_getd("problem", "gamma_shock1");
  Real z_shock1 = par_getd("problem", "z_shock1");
  // shock corrugation parameters: sin terms
  Real corr_sin1_A = par_getd("problem", "corr_sin1_A");
  Real corr_sin1_L = par_getd("problem", "corr_sin1_L");
  Real corr_sin1_f = 2.*M_PI / corr_sin1_L;
  Real corr_sin1_Lz = par_getd("problem", "corr_sin1_Lz");
  if (corr_sin1_Lz == 0) corr_sin1_Lz = HUGE_NUMBER;
  Real corr_sin1_fz = 2.*M_PI / corr_sin1_Lz;
  Real corr_sin2_A = par_getd("problem", "corr_sin2_A");
  Real corr_sin2_L = par_getd("problem", "corr_sin2_L");
  Real corr_sin2_f = 2.*M_PI / corr_sin2_L;
  Real corr_sin2_Lz = par_getd("problem", "corr_sin2_Lz");
  if (corr_sin2_Lz == 0) corr_sin2_Lz = HUGE_NUMBER;
  Real corr_sin2_fz = 2.*M_PI / corr_sin2_Lz;
  Real corr_sin3_A = par_getd("problem", "corr_sin3_A");
  Real corr_sin3_L = par_getd("problem", "corr_sin3_L");
  Real corr_sin3_f = 2.*M_PI / corr_sin3_L;
  Real corr_sin3_Lz = par_getd("problem", "corr_sin3_Lz");
  if (corr_sin3_Lz == 0) corr_sin3_Lz = HUGE_NUMBER;
  Real corr_sin3_fz = 2.*M_PI / corr_sin3_Lz;
  // shock corrugation parameters: cos terms
  Real corr_cos1_A = par_getd("problem", "corr_cos1_A");
  Real corr_cos1_L = par_getd("problem", "corr_cos1_L");
  Real corr_cos1_f = 2.*M_PI / corr_cos1_L;
  Real corr_cos1_Lz = par_getd("problem", "corr_cos1_Lz");
  if (corr_cos1_Lz == 0) corr_cos1_Lz = HUGE_NUMBER;
  Real corr_cos1_fz = 2.*M_PI / corr_cos1_Lz;
  Real corr_cos2_A = par_getd("problem", "corr_cos2_A");
  Real corr_cos2_L = par_getd("problem", "corr_cos2_L");
  Real corr_cos2_f = 2.*M_PI / corr_cos2_L;
  Real corr_cos2_Lz = par_getd("problem", "corr_cos2_Lz");
  if (corr_cos2_Lz == 0) corr_cos2_Lz = HUGE_NUMBER;
  Real corr_cos2_fz = 2.*M_PI / corr_cos2_Lz;
  Real corr_cos3_A = par_getd("problem", "corr_cos3_A");
  Real corr_cos3_L = par_getd("problem", "corr_cos3_L");
  Real corr_cos3_f = 2.*M_PI / corr_cos3_L;
  Real corr_cos3_Lz = par_getd("problem", "corr_cos3_Lz");
  if (corr_cos3_Lz == 0) corr_cos3_Lz = HUGE_NUMBER;
  Real corr_cos3_fz = 2.*M_PI / corr_cos3_Lz;
  // magnetic field parameters for constant By
  Real bfield_A = par_getd("problem", "bfield_A");

  // initialize the grid conditions
  Real eta, rho_pert;
  Real rho, press, vel;
  Real enthalpy;

  vel = - gamma2v(gamma_shock1);
  Real _gamma = gamma_shock1;
  Real sqr_gamma, sqr_b;

  for (k = kl; k <= ku; k++) {
    for (j = jl; j <= ju; j++) {

      // read the current location
      cc_pos(pGrid,0,j,k,&z,&r,&x3);

      // radial profile
      eta = 0.5 + 0.5*cos(3.*r/rmax);

      // 0: AMBIENT MEDIUM
      rho = eta * (rho_jet - rho_amb) + rho_amb;
      press = eta * (press_jet - press_amb) + press_amb;

      for (i = iu; i <= iu+nghost; i++) {

        // read the current location
        cc_pos(pGrid,i,j,k,&z,&r,&x3);

        // calculate density corrugation
        rho_pert = corr_sin1_A * sin(corr_sin1_f * r + corr_sin1_fz * (z-z_shock1-vel*time))
                 + corr_cos1_A * cos(corr_cos1_f * r + corr_cos1_fz * (z-z_shock1-vel*time))
                 + corr_sin2_A * sin(corr_sin2_f * r + corr_sin2_fz * (z-z_shock1-vel*time))
                 + corr_cos2_A * cos(corr_cos2_f * r + corr_cos2_fz * (z-z_shock1-vel*time))
                 + corr_sin3_A * sin(corr_sin3_f * r + corr_sin3_fz * (z-z_shock1-vel*time))
                 + corr_cos3_A * cos(corr_cos3_f * r + corr_cos3_fz * (z-z_shock1-vel*time));
        rho = (eta * (rho_jet - rho_amb) + rho_amb) * (1.0 + rho_pert);

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
        pGrid->U[k][j][i].B2c = bfield_A;
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
        pGrid->B2i[k][j][i] = bfield_A;
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
