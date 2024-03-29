#include "copyright.h"
/*============================================================================*/
/*! \file IntSh2.c
 *  \brief Problem generator for the corrugated internal shock collision simulations
 *  in context of relativistic jets in microquasars.
 *  Author: Patryk Pjanka, Nordita, 2021 */
/*============================================================================*/

//#define DEBUG_RHCOND

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#ifdef PARTICLES
#include "particles/particle.h"
#endif

#include <gsl/gsl_poly.h> // NOTE: -lgsl -lgslcblas -lm needed at linking stage

#ifndef M_PI // For whatever reason not found in debug mode...
#define M_PI 3.1415926535897932384626433832795028841971
#endif

// C implementation of vectors of pairs
#define TYPE int
#define TYPED_NAME(x) int_##x
#include "prob/vectors.h"
#undef TYPE
#undef TYPED_NAME
#define TYPE Real
#define TYPED_NAME(x) Real_##x
#include "prob/vectors.h"
#undef TYPE
#undef TYPED_NAME

#define RANDOM_DIM 2 // dimensionality of velocity space for drawing random directions

// variables global to this file
static int n_shocks = 2;
#ifdef PARTICLES
static Real shock_detection_threshold;
static Real min_sin_angle = 1./sqrt(3);
char name[50];

// PARTICLE INJECTION DISTRIBUTIONS

static int injection_time_type = 0;
// injection_time_type == 0 -- no particle injection, shock tracking only
// injection_time_type == 1 -- all at once, shock by shock
static Real* injection_time;
// injection_time_type == 2 -- gaussian, shock by shock
static Real* injection_time_sigma;

// pointer to a wrapper function that will set particle injection velocity
static void (*draw_particle_vel) (Real time, int sh, Real* v1, Real* v2, Real* v3);

static int injection_type = 1; // 1 -- separable energy / direction distributions

// particle injection with separable energy / direction distributions
static void draw_particle_vel_separable (Real time, int sh, Real* v1, Real* v2, Real* v3);
static void (*draw_particle_vel_separable_value) (Real time, int sh, Real* vel);
static void (*draw_particle_vel_separable_dir) (Real time, int sh, Real* vel, Real* v1, Real* v2, Real* v3);

// [Separable distr.] injection energy
static int injection_en_type = 1;
// injection_energy_type == 1 -- all particles at the same energy, random angle
static void draw_particle_vel_separable_value_type1 (Real time, int sh, Real* vel);
// injection_energy_type == 2 -- gaussian distr. in energy, random angle
static void draw_particle_vel_separable_value_type2 (Real time, int sh, Real* vel);
// injection_energy_type - related variables
static Real* injection_vel;
static Real* injection_gamma;
static Real* injection_gamma_sigma;

// [Separable distr.] injection direction
static int injection_dir_type = 1;
// injection_dir_type == 1 -- random direction
static void draw_particle_vel_separable_dir_type1 (Real time, int sh, Real* vel, Real* v1, Real* v2, Real* v3);
// injection_dir_type == 2 -- elongated in m directions, PDF ~ 1+A*sin(m*(phi-phi0)+pi/2)
static void draw_particle_vel_separable_dir_type2 (Real time, int sh, Real* vel, Real* v1, Real* v2, Real* v3);
static Real* injection_A;
static int* injection_m;
static Real* injection_phi0;

#endif // PARTICLES

//----------------------------------------------------------------------------------

static inline Real gaussianCDF (Real x, Real x0, Real sigma) {
  return 0.5 * (1.0 + erf((x-x0)/(sigma*sqrt(2.0))));
}

// Inverse CDF of a gaussian distribution
// Credits: Peter John Acklam (https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/)
// Direct source: https://stackoverflow.com/questions/27830995/inverse-cumulative-distribution-function-in-c
static double gaussianInverseCDF (double p) {

  double a1 = -39.69683028665376;
  double a2 = 220.9460984245205;
  double a3 = -275.9285104469687;
  double a4 = 138.3577518672690;
  double a5 =-30.66479806614716;
  double a6 = 2.506628277459239;

  double b1 = -54.47609879822406;
  double b2 = 161.5858368580409;
  double b3 = -155.6989798598866;
  double b4 = 66.80131188771972;
  double b5 = -13.28068155288572;

  double c1 = -0.007784894002430293;
  double c2 = -0.3223964580411365;
  double c3 = -2.400758277161838;
  double c4 = -2.549732539343734;
  double c5 = 4.374664141464968;
  double c6 = 2.938163982698783;

  double d1 = 0.007784695709041462;
  double d2 = 0.3224671290700398;
  double d3 = 2.445134137142996;
  double d4 = 3.754408661907416;

  //Define break-points.
  double p_low =  0.02425;
  double p_high = 1 - p_low;
  long double  q, r, e, u;
  long double x = 0.0;

  //Rational approximation for lower region.
  if (0 < p && p < p_low) {
      q = sqrt(-2*log(p));
      x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
  }

  //Rational approximation for central region.
  if (p_low <= p && p <= p_high) {
      q = p - 0.5;
      r = q*q;
      x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
  }

  //Rational approximation for upper region.
  if (p_high < p && p < 1) {
      q = sqrt(-2*log(1-p));
      x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
  }

  //Pseudo-code algorithm for refinement
  if(( 0 < p)&&(p < 1)){
      e = 0.5 * erfc(-x/sqrt(2)) - p;
      u = e * sqrt(2*M_PI) * exp(x*x/2);
      x = x - u/(1 + x*u/2);
  }

  return x;
}

static inline Real draw_random_gaussian (Real x0, Real sigma) {
  Real draw = gaussianInverseCDF(rand() * 1.0 / RAND_MAX);
  return draw*sigma + x0;
}

static inline Real dirDistrCDF (Real phi, Real m, Real A) {
  return (1. / (2.*M_PI)) * (phi + (A/m)*(1.-cos(m*phi)));
}
static Real dirDistrInverseCDF (Real x, Real m, Real A, Real phi0) {
  static Real precision=0.01;
  // just a simple numerical inversion here, note that the CDF is monotonic
  Real l=0., r=2*M_PI, mid;
  while (r-l > precision) {
    mid = 0.5*(l+r);
    if ((dirDistrCDF(l,m,A)-x) * (dirDistrCDF(mid,m,A)-x) <= 0) {
      r = mid;
    } else {
      l = mid;
    }
  }
  return 0.5*(l+r) + phi0 - M_PI/(2.*m);
}

static inline Real draw_random_dirDistr (Real m, Real A, Real phi0) {
  return dirDistrInverseCDF(rand() * 1.0 / RAND_MAX, m,A,phi0);
}

//----------------------------------------------------------------------------------

// Special Relativity handling functions
static inline Real v2gamma (Real vel)
{return 1./sqrt(1.-vel*vel);}
static inline Real gamma2v (Real _gamma)
{return sqrt(1.-1./(_gamma*_gamma));}
static inline Real dv2dgamma(Real vel, Real dv)
{return dv * vel/pow(1.0-SQR(vel),1.5);}
static inline Real dgamma2dv (Real gamma, Real dgamma)
{return dgamma / (SQR(gamma)*sqrt(SQR(gamma)-1.0));}

// declarations
void inflow_boundary (GridS *pGrid);
#ifdef PARTICLES
static bool part_in_rank (const Real3Vect pos); // end of file
void outflow_particle(GridS *pG); // bvals_particle.c (remove static!)
#endif

//----------------------------------------------------------------------------------

// Solver for the post-shock conditions from relativistic Rankine-Hugoniot conditions
// -- post-shock Lorentz factor space, with Bfield included
void postshock_gamma_bfield (Real adiab_idx,
		Real rho1, Real press1, Real vel1, Real gamma1, Real B1,
		Real* rho2, Real* press2, Real* vel2, Real* gamma2, Real* B2) {

  #ifdef DEBUG_RHCOND
  printf("gamma1 = %.2e\n", gamma1);
  #endif

  // calculate pre-shock constants
  Real D = B1 * vel1; // magnetic flux
  Real A = gamma1 * rho1 * vel1; // effective mass flux
  Real B = (rho1 + adiab_idx * press1 / (adiab_idx - 1.0)) * SQR(gamma1*vel1)
		  + press1 + SQR(D) * (1.0 + 0.5 / SQR(vel1*gamma1) ); // momentum flux
  Real C = (rho1 + adiab_idx * press1 / (adiab_idx - 1.0)) * SQR(gamma1) * vel1
		  + SQR(D) / vel1; // energy flux

  #ifdef DEBUG_RHCOND
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

  #ifdef DEBUG_RHCOND
  printf("a6 = %.2e, a5 = %.2e, a4 = %.2e, a3 = %.2e, a2 = %.2e, a1 = %.2e, a0 = %.2e\n", a6, a5, a4, a3, a2, a1, a0);
  #endif

  // solve the quartic equation for vel2
  Real a[7] = {a0,a1,a2,a3,a4,a5,a6};
  Real z[12]; // solutions
  gsl_poly_complex_workspace* w = gsl_poly_complex_workspace_alloc (7);
  gsl_poly_complex_solve (a, 7, w, z);
  gsl_poly_complex_workspace_free (w);

  Real gamma2_here, vel2_here, rho2_here, press2_here, B2_here;

  #ifdef DEBUG_RHCOND
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

  #ifdef DEBUG_RHCOND
  printf("gamma2 = %.2e, vel2 = %.2e, rho2 = %.2e, press2 = %.2e, B2 = %.2e\n", *gamma2, *vel2, *rho2, *press2, *B2);
  //exit(0);
  #endif

}

/*----------------------------------------------------------------------------*/
/* problem:  */

void problem(DomainS *pDomain)
{
  // initialize the random number generator
  time_t t;
  srand((unsigned) (time(&t) + myID_Comm_world));

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
	// shock detection parameters
	shock_detection_threshold = par_getd("problem", "shock_detection_threshold");

  Real eta, enthalpy;
  Real rho, press, vel, B, _gamma;
  Real rho1, press1, vel1, B1, gamma1;
  Real sqr_b, sqr_gamma;

  // INITIALIZE THE MHD GRID -----------------------------------------------------
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
          #ifdef DEBUG_RHCOND
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
          #ifdef DEBUG_RHCOND
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
    #ifdef PARTICLES
    set_bvals_particle_fun(right_x1, outflow_particle);
    #endif
  }

  #ifdef PARTICLES
  // INITIALIZE THE PARTICLES -----------------------------------------------------
  // initialize particle properties for each type
  // particle stopping time, sim.u., obsolete here
  tstop0[0] = par_getd_def("particle","tstop",1.0e20);
  // charge-to-mass ratio, q/mc, see Mignone et al. (2018), eq. 18
  grproperty[0].alpha = par_getd_def("particle", "alpha", 0.0);

  long int p, pgrid;
  int npart = par_geti("particle", "parnumgrid");
  pGrid->nparticle = 0; pgrid = 0;
  // Figure out how to split particles between shocks and cells
  int npart_shock [2]; int part_jstep [2]; int npart_per_j [2];
  Real z_shock [2] = {z_shock1, z_shock2};
  int sh, ibuff, npart_tot = 0;
  char sbuff [50];
  for (sh = 0; sh < n_shocks; sh++) {
    snprintf(sbuff, 50, "fpart_shock%d", sh+1);
    npart_shock[sh] = (int) (npart * par_getd("problem", sbuff));
    ibuff = (int) npart_shock[0]/pGrid->Nx[1];
    if (ibuff > 1) {
      npart_per_j[sh] = ibuff;
      part_jstep[sh] = 1;
    } else {
      npart_per_j[sh] = 1;
      part_jstep[sh] = (int) pGrid->Nx[1]/npart_shock[sh];
    }
    npart_shock[sh] = (int) npart_per_j[sh] * pGrid->Nx[1] / part_jstep[sh];
    npart_tot += npart_shock[sh];
  }
  if (npart_tot != npart) {
    printf("The total number of injected particles does not match the parnumgrid given. Please ensure that either:\n 1) The number of particles for each shock is divisible by the number of cells in the x2 direction.\n OR\n 2) The number of cells in the x2 direction is divisible by the number of particles for each shock.\n");
    printf(" -- Current setup -- \n");
    for (sh = 0; sh < n_shocks; sh++) {
      printf("  > Shock%i:\n", sh+1);
      printf("     npart_shock = %i\n", npart_shock[sh]);
      printf("     npart_per_j = %i\n", npart_per_j[sh]);
      printf("     part_jstep  = %i\n", part_jstep[sh]);
    }
    printf("\n");
    exit(0);
  }
  // Set up the particles
  Real x1l, x1u;
  Real3Vect pos;
  for (k = ks; k <= ke; k++) {
    for (sh = 0; sh < n_shocks; sh++) {
      for (i = ie; i >= is-1; i--) {
        // read the current location
        fc_pos(pGrid,i  ,j,k,&x1l,&r,&x3);
        fc_pos(pGrid,i+1,j,k,&x1u,&r,&x3);
        if ((x1l <= z_shock[sh]) && (z_shock[sh] < x1u)) {
          for (j = js; j <= je+1; j+=part_jstep[sh]) {

            // read the current location
            fc_pos(pGrid,i  ,j,k,&x1l,&r,&x3);
            fc_pos(pGrid,i+1,j,k,&x1u,&r,&x3);
            cc_pos(pGrid,i  ,j,k,&z  ,&r,&x3);

            // add the particles
            for (p = 0; p < npart_per_j[sh]; p++) {
              pos.x1 = z_shock[sh];
              pos.x2 = r;
              pos.x3 = x3;
              if (part_in_rank(pos)) {
                (pGrid->nparticle)++;
                if (pGrid->nparticle+2 > pGrid->arrsize) {
                  particle_realloc(pGrid, pGrid->nparticle+2);
                }
                // particle properties
                pGrid->particle[pgrid].property = 0;
                pGrid->particle[pgrid].x1 = pos.x1;
                pGrid->particle[pgrid].x2 = pos.x2;
                pGrid->particle[pgrid].x3 = pos.x3;
                pGrid->particle[pgrid].v1 = 0.;
                pGrid->particle[pgrid].v2 = 0.;
                pGrid->particle[pgrid].v3 = 0.;
                pGrid->particle[pgrid].pos = 1; /* grid particle */
                // NOTE: my_id is NOT unique across processors / grids
                //  - The unique particle ID is the (init_id, my_id) pair
                pGrid->particle[pgrid].my_id = pgrid;
                pGrid->particle[pgrid].shock_of_origin = sh;
                pGrid->particle[pgrid].injected = 0;
                #ifdef MPI_PARALLEL
                pGrid->particle[pgrid].init_id = myID_Comm_world;
                #endif
                pgrid++;
              } // part_in_rank
            } // p
          } // j
          sh += 1; // if IF is passed, the shock is done, proceed to the next one from the current i-index
        } // if z_shock inside i'th cell
      } // i
    } // sh
  } // k

  // set up particle injection variables
  char buf [30];
  // Injection as a function of time
  injection_time_type = par_geti_def("problem", "injection_time_type", 0);
  // 0 -> no injection
  if (injection_time_type == 1) { // all at once, shock by shock
    injection_time = malloc(n_shocks * sizeof(Real));
    for (sh = 0; sh < n_shocks; sh++) {
      sprintf(buf, "injection_time_sh%i", sh+1);
      injection_time[sh] = par_getd("problem", buf);
    }
  } else if (injection_time_type == 2) { // gaussian, shock by shock
    injection_time = malloc(n_shocks * sizeof(Real));
    injection_time_sigma = malloc(n_shocks * sizeof(Real));
    for (sh = 0; sh < n_shocks; sh++) {
      sprintf(buf, "injection_time_sh%i", sh+1);
      injection_time[sh] = par_getd("problem", buf);
      sprintf(buf, "injection_time_sigma_sh%i", sh+1);
      injection_time_sigma[sh] = par_getd("problem", buf);
    }
  }

  // Energy / direction distributions at injection
  injection_type = par_geti_def("problem", "injection_type", 1);
  if (injection_type == 1) { // separable energy / direction distr.
    draw_particle_vel = &draw_particle_vel_separable;
    // energy distribution setup
    injection_en_type = par_geti_def("problem", "injection_en_type", 1);
    if (injection_en_type == 1) { // single velocity
      draw_particle_vel_separable_value = &draw_particle_vel_separable_value_type1;
      injection_vel = malloc(n_shocks * sizeof(Real));
      for (sh = 0; sh < n_shocks; sh++) {
        sprintf(buf, "injection_vel_sh%i", sh+1);
        injection_vel[sh] = par_getd("problem", buf);
      }
    } else if (injection_en_type == 2) { // gaussian in energy
      draw_particle_vel_separable_value = &draw_particle_vel_separable_value_type2;
      injection_gamma = malloc(n_shocks * sizeof(Real));
      injection_gamma_sigma = malloc(n_shocks * sizeof(Real));
      for (sh = 0; sh < n_shocks; sh++) {
        sprintf(buf, "injection_gamma_sh%i", sh+1);
        injection_gamma[sh] = par_getd("problem", buf);
        sprintf(buf, "injection_gamma_sigma_sh%i", sh+1);
        injection_gamma_sigma[sh] = par_getd("problem", buf);
      }
    }
    // direction distribution setup
    injection_dir_type = par_geti_def("problem", "injection_dir_type", 1);
    if (injection_dir_type == 1) { // random direction
      draw_particle_vel_separable_dir = &draw_particle_vel_separable_dir_type1;
    } else if (injection_dir_type == 2) { // elongated in m directions, PDF ~ 1+A*sin(m*(phi-phi0)+pi/2)
      draw_particle_vel_separable_dir = &draw_particle_vel_separable_dir_type2;
      injection_A = malloc(n_shocks * sizeof(Real));
      injection_m = malloc(n_shocks * sizeof(int));
      injection_phi0 = malloc(n_shocks * sizeof(Real));
      for (sh = 0; sh < n_shocks; sh++) {
        sprintf(buf, "injection_A_sh%i", sh+1);
        injection_A[sh] = par_getd("problem", buf);
        sprintf(buf, "injection_m_sh%i", sh+1);
        injection_m[sh] = par_geti("problem", buf);
        sprintf(buf, "injection_phi0_sh%i", sh+1);
        injection_phi0[sh] = par_getd("problem", buf);
      }
    }
  }

  #endif // PARTICLES
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
  // write the global variables

  fwrite(name, sizeof(char),50,fp);
  fwrite(&n_shocks, sizeof(int),1,fp);
  fwrite(&shock_detection_threshold, sizeof(Real),1,fp);
  fwrite(&min_sin_angle, sizeof(Real),1,fp);

  #ifdef PARTICLES
  fwrite(&injection_time_type, sizeof(int),1,fp);
  if (injection_time_type == 1) { // all at once, shock by shock
    fwrite(injection_time, sizeof(Real),n_shocks,fp);
  } else if (injection_time_type == 2) { // gaussian, shock by shock
    fwrite(injection_time, sizeof(Real),n_shocks,fp);
    fwrite(injection_time_sigma, sizeof(Real),n_shocks,fp);
  }
  fwrite(&injection_type, sizeof(int),1,fp);
  if (injection_type == 1) { // separable energy / direction distributions
    fwrite(&injection_en_type, sizeof(int),1,fp);
    if (injection_en_type == 1) { // single velocity, random direction
      fwrite(injection_vel, sizeof(Real),n_shocks,fp);
    } else if (injection_en_type == 2) { // gaussian in energy, random direction
      fwrite(injection_gamma, sizeof(Real),n_shocks,fp);
      fwrite(injection_gamma_sigma, sizeof(Real),n_shocks,fp);
    }
    fwrite(&injection_dir_type, sizeof(int),1,fp);
    if (injection_dir_type == 2) { // elongated in m directions, PDF ~ 1+A*sin(m*(phi-phi0)+pi/2)
      fwrite(injection_A, sizeof(Real),n_shocks,fp);
      fwrite(injection_m, sizeof(int),n_shocks,fp);
      fwrite(injection_phi0, sizeof(Real),n_shocks,fp);
    }
  }
  #endif

  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  // read the global variables

  char buffer[2048]; int buffer_length = 0;
  buffer_length += sprintf(buffer+buffer_length, "[Proc %i] Reading problem data from restart file:\n", myID_Comm_world);

  fread(name, sizeof(char),50,fp);
  fread(&n_shocks, sizeof(int),1,fp);
  fread(&shock_detection_threshold, sizeof(Real),1,fp);
  fread(&min_sin_angle, sizeof(Real),1,fp);

  buffer_length += sprintf(buffer+buffer_length, "  n_shocks = %i\n  shock_detection_thr = %.2e\n  min_sin_angle = %.2e\n", n_shocks, shock_detection_threshold, min_sin_angle);

  #ifdef PARTICLES

  // injection times
  fread(&injection_time_type, sizeof(int),1,fp);
  buffer_length += sprintf(buffer+buffer_length, "  inj_time_type = %i\n", injection_time_type);
  if (injection_time_type == 1) { // all at once, shock by shock
    injection_time = malloc(n_shocks * sizeof(Real));
    fread(injection_time, sizeof(Real),n_shocks,fp);
    for (int sh = 0; sh < n_shocks; sh++) {
      buffer_length += sprintf(buffer+buffer_length, "    inj_time[%i] = %.2f\n", sh, injection_time[sh]);
    }
  } else if (injection_time_type == 2) { // gaussian, shock by shock
    injection_time = malloc(n_shocks * sizeof(Real));
    injection_time_sigma = malloc(n_shocks * sizeof(Real));
    fread(injection_time, sizeof(Real),n_shocks,fp);
    fread(injection_time_sigma, sizeof(Real),n_shocks,fp);
    for (int sh = 0; sh < n_shocks; sh++) {
      buffer_length += sprintf(buffer+buffer_length, "    inj_time[%i] = %.2f\n", sh, injection_time[sh]);
      buffer_length += sprintf(buffer+buffer_length, "    inj_time_sigma[%i] = %.2f\n", sh, injection_time_sigma[sh]);
    }
  }

  // injection energy / direction distributions
  fread(&injection_type, sizeof(int),1,fp);
  buffer_length += sprintf(buffer+buffer_length, "  inj_type = %i\n", injection_type);
  if (injection_type == 1) { // separable distributions
    draw_particle_vel = &draw_particle_vel_separable;
    // energy distribution setup
    fread(&injection_en_type, sizeof(int),1,fp);
    buffer_length += sprintf(buffer+buffer_length, "  inj_en_type = %i\n", injection_en_type);
    if (injection_en_type == 1) { // single velocity, random direction
      draw_particle_vel_separable_value = &draw_particle_vel_separable_value_type1;
      injection_vel = (Real*) malloc(n_shocks * sizeof(Real));
      fread(injection_vel, sizeof(Real),n_shocks,fp);
      for (int sh = 0; sh < n_shocks; sh++) {
        buffer_length += sprintf(buffer+buffer_length, "    inj_vel[%i] = %.2f\n", sh, injection_vel[sh]);
      }
    } else if (injection_en_type == 2) { // gaussian energy, random direction
      draw_particle_vel_separable_value = &draw_particle_vel_separable_value_type2;
      injection_gamma = (Real*) malloc(n_shocks * sizeof(Real));
      injection_gamma_sigma = (Real*) malloc(n_shocks * sizeof(Real));
      fread(injection_gamma, sizeof(Real),n_shocks,fp);
      fread(injection_gamma_sigma, sizeof(Real),n_shocks,fp);
      for (int sh = 0; sh < n_shocks; sh++) {
        buffer_length += sprintf(buffer+buffer_length, "    inj_gamma[%i] = %.2f\n", sh, injection_gamma[sh]);
        buffer_length += sprintf(buffer+buffer_length, "    inj_gamma_sigma[%i] = %.2f\n", sh, injection_gamma_sigma[sh]);
      }
    }
    // direction distribution setup
    fread(&injection_dir_type, sizeof(int),1,fp);
    if (injection_dir_type == 1) { // random direction
      draw_particle_vel_separable_dir = &draw_particle_vel_separable_dir_type1;
    } else if (injection_dir_type == 2) { // // elongated in m directions, PDF ~ 1+A*sin(m*(phi-phi0)+pi/2)
      draw_particle_vel_separable_dir = &draw_particle_vel_separable_dir_type2;
      injection_A = malloc(n_shocks * sizeof(Real));
      injection_m = malloc(n_shocks * sizeof(int));
      injection_phi0 = malloc(n_shocks * sizeof(Real));
      fread(injection_A, sizeof(Real),n_shocks,fp);
      fread(injection_m, sizeof(int),n_shocks,fp);
      fread(injection_phi0, sizeof(Real),n_shocks,fp);
      for (int sh = 0; sh < n_shocks; sh++) {
        buffer_length += sprintf(buffer+buffer_length, "    inj_A[%i] = %.2f\n", sh, injection_A[sh]);
        buffer_length += sprintf(buffer+buffer_length, "    inj_m[%i] = %i\n", sh, injection_m[sh]);
        buffer_length += sprintf(buffer+buffer_length, "    inj_phi0[%i] = %.2f\n", sh, injection_A[sh]);
      }
    }
  }
  #endif
  buffer_length += sprintf(buffer+buffer_length, "all problem data read from restart file.\n");

  printf("%s\n", buffer);

  // initialize the random number generator
  time_t t;
  srand((unsigned) (time(&t) + myID_Comm_world));

  // enroll the bvals functions
  int nl, nd;
  Real z,r,x3;
  GridS* grid;
  for (nl=0; nl<(pM->NLevels); nl++){
    for (nd=0; nd<(pM->DomainsPerLevel[nl]); nd++){
      grid = pM->Domain[nl][nd].Grid;
      if (grid != NULL){
        fc_pos(grid,grid->ie+1,0,0,&z,&r,&x3);
        if ((z - par_getd("domain1", "x1max")) < 1.0e-6) {
          bvals_mhd_fun(&(pM->Domain[nl][nd]), right_x1, inflow_boundary);
          #ifdef PARTICLES
          set_bvals_particle_fun(right_x1, outflow_particle);
          #endif
        }
      }
    }
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

#ifdef PARTICLES
PropFun_t get_usr_par_prop(const char *name)
{
  return NULL;
}

void gasvshift(const Real x1, const Real x2, const Real x3,
                                    Real *u1, Real *u2, Real *u3)
{
  return;
}

void Userforce_particle(Real3Vect *ft, const Real x1, const Real x2, const Real x3,
                                    const Real v1, const Real v2, const Real v3)
{
  return;
}
#endif

void Userwork_in_loop(MeshS *pM)
{
  #ifdef PARTICLES
  // LOCALIZE THE SHOCKS WITHIN THIS PROCESSOR --------------------------------
  int i,j, nl,nd,ng, ilen, jlen;
  GridS* grid;
  // 0) find out what grids are on this processor..
  static int n_local_grids;
  static int** local_grids;
  if (n_local_grids == 0) { // initialize on first use
    struct int_vector local_grids_vec = int_vector_default;
    for (nl=0; nl<(pM->NLevels); nl++){
      for (nd=0; nd<(pM->DomainsPerLevel[nl]); nd++){
        if (pM->Domain[nl][nd].Grid != NULL){
          int_append_to_vector(&local_grids_vec, nl,nd);
        }
      }
    }
    n_local_grids = local_grids_vec.n_elements;
    local_grids = int_vector_to_array(&local_grids_vec);
  }

  // prepare arrays to hold gradient values
  static Real*** gradient;
  if (gradient == NULL) { // initialize on first use
    gradient = (Real***) calloc_1d_array (n_local_grids, sizeof(Real**));
    for (ng = 0; ng < n_local_grids; ng++) {
      nl = local_grids[ng][0];
      nd = local_grids[ng][1];
      grid = pM->Domain[nl][nd].Grid;
      ilen = grid->Nx[0] + nghost - 1;
      jlen = grid->Nx[1] + nghost - 1;
      gradient[ng] = (Real**) calloc_2d_array (jlen,ilen, sizeof(Real));
    }
  }
  // 1) calculate gradient = (d(IVX)/dx)^2 + (d(IVX)/dy)^2
  Real centroid, norm, sin_angle;
  int il, iu, sh, idx;
  Real x1l, x1r, x2, x3, dist_x1, dist_x1_min, x1len;
  struct Real_vector *centroids;
  struct Real_vector_element *Real_elem;
  struct int_vector idx_max_value = int_vector_default;
  struct int_vector_element *int_elem;
  int consecutive; Real current_max_value;
  long int p;
  bool move;
  for (ng = 0; ng < n_local_grids; ng++) {
    nl = local_grids[ng][0];
    nd = local_grids[ng][1];
    grid = pM->Domain[nl][nd].Grid;
    ilen = grid->Nx[0] + nghost - 1;
    jlen = grid->Nx[1] + nghost - 1;
    x1len = grid->MaxX[0] - grid->MinX[0];

    // initialize centroids for this grid
    centroids = (struct Real_vector*) malloc((jlen-1) * sizeof(struct Real_vector));
    for (j = 1; j < jlen-1; j++) {
      centroids[j] = Real_vector_default;
    }

    // localize shocks
    for (j = 1; j < jlen-1; j++) {
      consecutive = 0;
      idx_max_value = int_vector_default;
      for (i = 1; i < ilen-1; i++) {
        gradient[ng][j][i] =
              SQR((grid->Whalf[0][j][i+1].V1 - grid->Whalf[0][j][i].V1) / (grid->dx1))
            + SQR((grid->Whalf[0][j+1][i].V1 - grid->Whalf[0][j][i].V1) / (grid->dx2));
  // 2) if gradient above threshold, find local maxima and fit centroids
        if (gradient[ng][j][i] > shock_detection_threshold) {
          if (consecutive == 0) {
            int_append_to_vector(&idx_max_value, i, 0);
            current_max_value = gradient[ng][j][i];
          } else if (gradient[ng][j][i] > current_max_value) {
            idx_max_value.last_element->first = i;
          }
          consecutive++;
        } else if (consecutive > 0) {
          // ensure that the indices aren't too close to the ghost zones
          if (   idx_max_value.last_element->first < 2
              || idx_max_value.last_element->first > ilen-3) {
            int_drop_from_vector(&idx_max_value);
          }
          // reset consecutive counter
          consecutive = 0;
        }
      } // i
      if (consecutive > 0) {
        // ensure that the indices aren't too close to the ghost zones
        if (   idx_max_value.last_element->first < 2
            || idx_max_value.last_element->first > ilen-3) {
          int_drop_from_vector(&idx_max_value);
        }
      }
      if (idx_max_value.n_elements > 0) {
        for (int_elem = idx_max_value.first_element;
             int_elem != NULL;
             int_elem = int_elem->next) {
          idx = int_elem->first;
          // calculate the centroid
          centroid = 0; norm = 0;
          for (i = idx-1; i <= idx+1; i++) {
            fc_pos(grid, i+1,j,0, &x1l,&x2,&x3);
            centroid += gradient[ng][j][i] * x1l;
            norm     += gradient[ng][j][i];
          }
          centroid /= norm;
          // if the centroid does not fall in the ghost zones, save the relevant data
          fc_pos(grid, grid->is,j,0, &x1l,&x2,&x3);
          fc_pos(grid, grid->ie,j,0, &x1r,&x2,&x3);
          if (centroid > x1l && centroid < x1r) {
            // update idx
            fc_pos(grid, idx,j,0, &x1l,&x2,&x3);
            if (centroid < x1l) {
              idx -= 1;
            } else {
              fc_pos(grid, idx+1,j,0, &x1r,&x2,&x3);
              if (centroid > x1r) idx += 1;
            }
            // calculate angle (abs(dq/dx) / sqrt(gradient))
            i = idx;
            sin_angle = fabs((grid->Whalf[0][j][i+1].V1 - grid->Whalf[0][j][i].V1) / (grid->dx1)) / sqrt(gradient[ng][j][i]);
            sin_angle = fmax(sin_angle, min_sin_angle);
            Real_append_to_vector(&(centroids[j]), centroid, sin_angle);
          }
        }
        int_clear_vector(&idx_max_value);
      }
    } // j

    // UPDATE PARTICLES ----------------------------------------------------------
    // 3) move uninitialized particles to their nearest shock's position
    for (p = 0; p < grid->nparticle; p++) {
      if (grid->particle[p].injected > 0) continue;
      // find out the j index corresponding to this particle
      j = (int) ((grid->particle[p].x2 - grid->MinX[1]) / (grid->dx2))
          + nghost/2;
      // check whether there is a shock nearby (~ within the particle's light cone) along the x1-dir
      if (grid->time > grid->dt) {
        dist_x1_min = 1.5*grid->dt;
      } else { // always move to the shock on first timestep
        dist_x1_min = x1len;
      }
      move = false;
      for (Real_elem = centroids[j].first_element;
           Real_elem != NULL;
           Real_elem = Real_elem->next) {
        dist_x1 = fabs(Real_elem->first - grid->particle[p].x1);
        if (dist_x1 < dist_x1_min) {
          dist_x1_min = dist_x1;
          centroid = Real_elem->first;
          move = true;
        }
      }
      // if there is a shock nearby (~ within the particle's light cone) along the x1-dir, move the particle there
      if (move) {
        if (centroid > grid->MinX[0] && centroid < grid->MaxX[0]) {
          grid->particle[p].shock_speed = (centroid - grid->particle[p].x1) / (grid->dt);
        }
        grid->particle[p].x1 = centroid;
      } else {
        // otherwise, advect the particle with its saved shock speed
        grid->particle[p].x1 += grid->particle[p].shock_speed * grid->dt;
      }
    } // p loop

    // clean up the centroids vector
    for (j = 1; j < jlen-1; j++) {
      Real_clear_vector(&(centroids[j]));
    }
    free(centroids);

    // 4) initialize particles if needed
    if (injection_time_type == 1) { // all at once
      for (sh = 0; sh < n_shocks; sh++) {
        if (   grid->time <= injection_time[sh]
            && grid->time+grid->dt > injection_time[sh]) {
          for (p = 0; p < grid->nparticle; p++) {
            if (grid->particle[p].shock_of_origin == sh
                && grid->particle[p].injected == 0) {
              /*#ifdef MPI_PARALLEL
              printf(" - [Proc %i] Injecting particle no (%i,%i) into shock %i.\n",
                  myID_Comm_world,
                  grid->particle[p].init_id, grid->particle[p].my_id, sh);
              #else // MPI off
              printf(" - Injecting particle no %i into shock %i.\n",
                  grid->particle[p].my_id, sh);
              #endif*/
              (*draw_particle_vel) (grid->time, sh,
                  &(grid->particle[p].v1),
                  &(grid->particle[p].v2),
                  &(grid->particle[p].v3));
              // mark particle as injected
              grid->particle[p].injected += 1;
            }
          }
        }
      }
    } else if (injection_time_type == 2) { // gaussian
      static Real injection_probability_in_tstep;
      for (sh = 0; sh < n_shocks; sh++) {
        for (p = 0; p < grid->nparticle; p++) {
          if (grid->particle[p].shock_of_origin == sh
              && grid->particle[p].injected == 0) {
            // apply temporal injection distribution
            injection_probability_in_tstep =
                gaussianCDF(grid->time+grid->dt,
                  injection_time[sh],injection_time_sigma[sh]);
            if (grid->time > 0) {
              injection_probability_in_tstep -= gaussianCDF(grid->time,
                  injection_time[sh],injection_time_sigma[sh]);
            }
            if ( (rand()*1.0/RAND_MAX) > injection_probability_in_tstep) {
              continue;
            }
            /*#ifdef MPI_PARALLEL
            printf(" - [Proc %i] Injecting particle no (%i,%i) into shock %i.\n",
                myID_Comm_world,
                grid->particle[p].init_id, grid->particle[p].my_id, sh);
            #else // MPI off
            printf(" - Injecting particle no %i into shock %i.\n",
                grid->particle[p].my_id, sh);
            #endif*/
            (*draw_particle_vel) (grid->time, sh,
                &(grid->particle[p].v1),
                &(grid->particle[p].v2),
                &(grid->particle[p].v3));
            // mark particle as injected
            grid->particle[p].injected += 1;
          }
        }
      }
    }
  } // ng loop
  #endif // PARTICLES
}

// particle injection functions

// wrapper for separable energy / direction distributions
static void draw_particle_vel_separable (Real time, int sh, Real* v1, Real* v2, Real* v3) {
  Real vel;
  draw_particle_vel_separable_value(time, sh, &vel);
  draw_particle_vel_separable_dir(time, sh, &vel, v1,v2,v3);
}

// separable energy distr. type1: const velocity
static void draw_particle_vel_separable_value_type1 (Real time, int sh, Real* vel) {
  (*vel) = injection_vel[sh];
}

// separable energy distr. type2: gaussian energy
static void draw_particle_vel_separable_value_type2 (Real time, int sh, Real* vel) {
  // draw particle energy
  static Real energy;
  static unsigned char safety;
  energy = 0.0;
  for (safety = 0; energy < 1.0 && safety < 128; safety++) {
    // we need at least mc^2 of energy
    energy = draw_random_gaussian(injection_gamma[sh], injection_gamma_sigma[sh]);
  }
  (*vel) = gamma2v(energy);
}

// separable direction distr. type1: random direction
static void draw_particle_vel_separable_dir_type1 (Real time, int sh, Real* vel, Real* v1, Real* v2, Real* v3) {
  // draw particle direction and apply
  #if RANDOM_DIM==2
  static Real phi;
  phi = (2.*M_PI * rand()) / RAND_MAX;
  (*v1) = (*vel) * cos(phi);
  (*v2) = (*vel) * sin(phi);
  (*v3) = 0.0;
  #elif RANDOM_DIM==3
  static Real theta, z;
  // draw a random 3D angle
  // -- using equal-area cylindrical projection, as described at https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
  theta = (2.*M_PI * rand()) / RAND_MAX;
  z = (2. * rand()) / RAND_MAX - 1.0;
  // select particle velocity
  (*v1) = (*vel) * sqrt(1.-z*z) * cos(theta);
  (*v2) = (*vel) * sqrt(1.-z*z) * sin(theta);
  (*v3) = (*vel) * z * cos(theta);
  #endif
}

// separable direction distr. type2 -- elongated in m directions (1+a*sin(phi*m/2)-shaped)
static void draw_particle_vel_separable_dir_type2 (Real time, int sh, Real* vel, Real* v1, Real* v2, Real* v3) {
  #if RANDOM_DIM!=2
  printf("[IntSh2-part.c] Error: Elongated injection direction distr. is currently only implemented for injection in 2D. Aborting.\n");
  exit(1);
  #endif
  Real phi = draw_random_dirDistr(injection_m[sh],injection_A[sh],injection_phi0[sh]);
  (*v1) = (*vel) * cos(phi);
  (*v2) = (*vel) * sin(phi);
  (*v3) = 0.0;
}

void Userwork_after_loop(MeshS *pM)
{
  // destroy the dynamically allocated global variables
  //  -- unnecessary, and disrupts restart functionality
  /*if (injection_time_type == 1) {
    free(injection_time);
  } else if (injection_time_type == 2 {
    free(injection_time);
    free(injection_time_sigma);
  }
  if (injection_en_type == 1) {
    free(injection_vel);
  } else if (injection_en_type == 2) {
    free(injection_gamma);
    free(injection_gamma_sigma);
  }
  if (injection_dir_type == 2 {
    free(injection_A);
    free(injection_m);
    free(injection_phi0);
  }
  */
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

      #pragma omp simd
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

/*=========================== PRIVATE FUNCTIONS ==============================*/
/*--------------------------------------------------------------------------- */

/*! \fn static int ParticleLocator(const RealVect pos)
 *  \brief Judge if the particle is in this MPI rank
 *  -- adapted from par_epicycle.c */
static bool part_in_rank (const Real3Vect pos)
{
  return ((pos.x1<x1upar) && (pos.x1>=x1lpar) && (pos.x2<x2upar)
      && (pos.x2>=x2lpar) &&(pos.x3<x3upar) && (pos.x3>=x3lpar));
}
