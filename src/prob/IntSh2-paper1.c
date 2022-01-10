#include "copyright.h"
/*============================================================================*/
/*! \file IntSh2-paper1.c
 *  \brief Problem generator for a corrugated shell collision problem
 *  , in context of the internal shock model in microquasars
 *
 * REFERENCE: Pjanka, Demidem, Veledina (2022), in prep. */
/*============================================================================*/

// #define DEBUG

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

// vector potentials for magnetic field adjustments
static Real vecPotLoops (Real x, Real y, Real norm, int n, Real x_1, Real x_2, Real y_1, Real y_2) {
  if (x > x_1 && x < x_2 && y > y_1 && y < y_2) {
    return norm
        * sin(M_PI*n*y/(-y_1 + y_2) - M_PI*n*y_1/(-y_1 + y_2))
        * cos(M_PI*x/(x_1 - x_2) + (1.0/2.0)*M_PI*(x_1 - 3*x_2)/(x_1 - x_2));
  } else {
    return 0.;
  }
}

// for corr_type = 4: try setting magnetic fields with a given bfield_sh and return the resulting mean magnetic energy density
static Real try_bfield (GridS *pGrid, Real corr_ampl, int n, Real x1, Real x2, Real bfield_sh) {
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  int k, ks = pGrid->ks, ke = pGrid->ke;
  Real dx = pGrid->dx1, dy = pGrid->dx2;
  Real y1 = par_getd("domain1", "x2min");
  Real y2 = par_getd("domain1", "x2max");
  Real x,y,x3;
  Real Benergy_tot = 0., surface_tot = 0.;
  Real norm = sqrt(corr_ampl) * bfield_sh * (x2-x1)/M_PI; // dx since Az is an integral quantity
  for (k = ks; k <= ke; k++) {
    for (j = js; j <= je+1; j++) {
      #pragma omp simd
      for (i = is; i <= ie+1; i++) {
        cc_pos(pGrid,i,j,k,&x,&y,&x3);
        // set Bz
        pGrid->B3i[k][j][i] = 0.;
        pGrid->U[k][j][i].B3c = 0.;
        // set bfield using the vector potential and centered finite difference derivatives
        if (x > x1 && x < x2) {
          pGrid->U[k][j][i].B1c = // Bx = dyAz
              (  vecPotLoops(x,y+0.5*dy, norm, n, x1,x2, y1,y2)
               - vecPotLoops(x,y-0.5*dy, norm, n, x1,x2, y1,y2))
              / dy;
          pGrid->U[k][j][i].B2c = // By = -dxAz + B0
            - (  vecPotLoops(x+0.5*dx,y, norm, n, x1,x2, y1,y2)
               - vecPotLoops(x-0.5*dx,y, norm, n, x1,x2, y1,y2))
              / dx
              + bfield_sh;
          Benergy_tot +=
              0.5 * ( SQR(pGrid->U[k][j][i].B1c) + SQR(pGrid->U[k][j][i].B2c));
          surface_tot += dx*dy;
        }
        if (i >= is) {
          fc_pos(pGrid,i,j,k,&x,&y,&x3);
          // test whether we are in the appropriate shell
          if (x > x1 && x < x2) {
            pGrid->B1i[k][j][i] = // Bx = dyAz
                (  vecPotLoops(x,y+0.5*dy, norm, n, x1,x2, y1,y2)
                 - vecPotLoops(x,y-0.5*dy, norm, n, x1,x2, y1,y2))
                / dy;
            pGrid->B2i[k][j][i] = // By = -dxAz
                - (  vecPotLoops(x+0.5*dx,y, norm, n, x1,x2, y1,y2)
                   - vecPotLoops(x-0.5*dx,y, norm, n, x1,x2, y1,y2))
                  / dx;
          }
        }
      }
    }
  }
  // total magnetic energy needs to be summed accross processors
  #ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &Benergy_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &surface_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  #endif
  Benergy_tot *= dx*dy / surface_tot; // gives energy density, to be compared with pressure / plasma_beta
  return Benergy_tot;
}

/*----------------------------------------------------------------------------*/
/* problem:  */

void problem(DomainS *pDomain)
{

  GridS *pGrid=(pDomain->Grid);
  int i, is = pGrid->is, ie = pGrid->ie;
  int j, js = pGrid->js, je = pGrid->je;
  int k, ks = pGrid->ks, ke = pGrid->ke;
  Real z,r,x1,x2,x3,x,y;

  // --- READ IN THE PROBLEM PARAMETERS ---

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
  #ifdef MHD
  Real sigmaB_sh [2] =
          {par_getd("problem", "sigmaB_sh1"), par_getd("problem", "sigmaB_sh2")};
  Real bfield_sh [2];
  Real B, sqr_b;
  #endif

  // --- PERFORM AUXILIARY PRE-LOOP CALCULATIONS ---

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
  // shell pressure
  Real press_sh [2];
  int press_set_mode = par_geti("problem", "press_set_mode");
  if (press_set_mode == 0) { // set pressure directly
    press_sh[0] = par_getd("problem", "press_sh1");
    press_sh[1] = par_getd("problem", "press_sh2");
  } else if (press_set_mode == 1) { // set plasma beta (requires bfield to be set!)
    press_sh[0] = par_getd("problem", "beta_sh1") * 0.5*SQR(bfield_sh[0]);
    press_sh[1] = par_getd("problem", "beta_sh2") * 0.5*SQR(bfield_sh[1]);
  }
  // corrugation
  int corr_switch = par_geti_def("problem", "corr_switch", 0);
  int corr_type = par_geti_def("problem", "corr_type", 1); // default=1 for backward compatibility
  Real corr_ampl = par_getd_def("problem", "corr_ampl", 0.0);
  int corr_nx = par_geti_def("problem", "corr_nx", 2);
  int corr_ny = par_geti_def("problem", "corr_ny", 2);
  // if corr_type == 4 (magnetic field varies within shells), pre-compute the magnetic fields first, in order to properly adjust pressures for fair comparison
  if (corr_type == 4 && corr_switch == 1) { // 4: magnetic field strength perturbations inside shells
    #ifdef MHD
    #ifdef DEBUG
    printf("Setting up magnetic field perturbations inside shells..\n");
    #endif
    Real dx = pGrid->dx1, dy = pGrid->dx2;
    Real y1 = par_getd("domain1", "x2min");
    Real y2 = par_getd("domain1", "x2max");
    Real norm, x1, x2, beta, Benergy_tot, buff;
    Real bfield_sh_old [2], Benergy_tot_old [2];
    int n = corr_ny; // number of loops in the vertical direction
    Real precision = 1.0e-9; // desired precision for mean Bfield energy vs plasma_beta matching
    int safety_counter;
    for(int sh=0; sh < 2; sh++) {
      #ifdef DEBUG
      printf(" -- SHELL %i\n", sh);
      printf("     original bfield_sh = %.2e\n", bfield_sh[sh]);
      #endif
      x1 = x1_sh[sh]; x2 = x2_sh[sh];
      if (sh == 0) {
        beta = par_getd("problem", "beta_sh1");
      } else {
        beta = par_getd("problem", "beta_sh2");
      }
      // initialize the parameter search
      bfield_sh_old[0] = 0.9 * bfield_sh[sh];
      Benergy_tot_old[0] = try_bfield(pGrid, corr_ampl, n, x1, x2, bfield_sh_old[0]);
      for (
            ;
            Benergy_tot_old[0] - (press_sh[sh] / beta) > 0.;
            bfield_sh_old[0] *= 0.9) {
        Benergy_tot_old[0] = try_bfield(pGrid, corr_ampl, n, x1, x2, bfield_sh_old[0]);
      }
      bfield_sh_old[1] = bfield_sh_old[0] + 0.1 * bfield_sh[sh];
      Benergy_tot_old[1] = try_bfield(pGrid, corr_ampl, n, x1, x2, bfield_sh_old[1]);
      for (
            ;
            Benergy_tot_old[1] - (press_sh[sh] / beta) < 0.;
            bfield_sh_old[1] *= 1.1) {
        Benergy_tot_old[1] = try_bfield(pGrid, corr_ampl, n, x1, x2, bfield_sh_old[1]);
      }
      if ( (Benergy_tot_old[0]-(press_sh[sh] / beta)) * (Benergy_tot_old[1]-(press_sh[sh] / beta)) > 0) {
        printf("   Bfield fitting failed to initialize for shell %i, aborting.\n", sh);
        exit(1);
      }
      bfield_sh[sh] = 0.5 * (bfield_sh_old[0] + bfield_sh_old[1]);
      // perform fitting
      safety_counter = 0;
      while ( (safety_counter++) < 1e3 ) { // 1000 iterations max
        #ifdef DEBUG
        printf("  - Iteration %i\n", safety_counter);
        printf("   bfield_sh_old = [%.2e,%.2e], bfield_sh = %.2e\n", bfield_sh_old[0], bfield_sh_old[1], bfield_sh[sh]);
        #endif
        Benergy_tot = try_bfield (pGrid, corr_ampl, n, x1, x2, bfield_sh[sh]);
        #ifdef DEBUG
        printf("   Benergy_tot_old = [%.2e,%.2e], Benergy_tot = %.2e\n", Benergy_tot_old[0], Benergy_tot_old[1], Benergy_tot);
        printf("   fabs(Benergy_tot / (press_sh[sh] / beta) - 1.0) = %.2e VS precision = %.2e\n", fabs(Benergy_tot / (press_sh[sh] / beta) - 1.0), precision);
        #endif
        // decide what to do to converge
        if (fabs(Benergy_tot / (press_sh[sh] / beta) - 1.0) <= precision) {
          #ifdef DEBUG
          printf("    Converged, calling break.\n");
          #endif
          break;
        } else {
          #ifdef DEBUG
          printf("    (Benergy_tot-(press_sh[sh] / beta)) = %.2e\n", (Benergy_tot-(press_sh[sh] / beta)));
          printf("    (Benergy_tot_old[0]-(press_sh[sh] / beta)) %.2e\n", (Benergy_tot_old[0]-(press_sh[sh] / beta)));
          printf("    (Benergy_tot_old[1]-(press_sh[sh] / beta)) %.2e\n", (Benergy_tot_old[1]-(press_sh[sh] / beta)));
          #endif
          if ( (Benergy_tot-(press_sh[sh] / beta)) * (Benergy_tot_old[0]-(press_sh[sh] / beta)) < 0 ) {
            buff = bfield_sh[sh];
            bfield_sh[sh] = 0.5 * (bfield_sh_old[0] + bfield_sh[sh]);
            bfield_sh_old[1] = buff;
            Benergy_tot_old[1] = Benergy_tot;
            #ifdef DEBUG
            printf("    - bfield_sh set to mean with left boundary bfields, i.e., %.2e\n", bfield_sh[sh]);
            #endif
          } else {
            buff = bfield_sh[sh];
            bfield_sh[sh] = 0.5 * (bfield_sh_old[1] + bfield_sh[sh]);
            bfield_sh_old[0] = buff;
            Benergy_tot_old[0] = Benergy_tot;
            #ifdef DEBUG
            printf("    - bfield_sh set to mean with right boundary bfields, i.e., %.2e\n", bfield_sh[sh]);
            #endif
          }
        }
      }
      if (safety_counter > 999) {
        printf("   Bfield fitting failed to converge for shell %i, aborting.\n", sh);
        exit(1);
      }
      #ifdef DEBUG
      printf(" -- SHELL %i done.\n\n", sh);
      #endif
    }
    #ifdef DEBUG
    printf("Magnetic field perturbations inside shells set.\n\n");
    #endif // debug
    #else
    printf("[IntSh2-paper1.c] ERROR: corr_type==4 requires MHD to take effect. Please reconfigure Athena --with-mhd and try again.\n");
    exit(1);
    #endif
  }

  // --- LOOP OVER ALL CELLS, SET ALL INITIAL CONDITIONS ---

  // set the initial conditions
  Real rho, press, vel, gamma, sqr_gamma, enthalpy;
	for (k = ks; k <= ke; k++) {
    for (j = js; j <= je+1; j++) {
      #pragma omp simd
      for (i = is; i <= ie+1; i++) {

        // read the current location
        cc_pos(pGrid,i,j,k,&z,&r,&x3);

        if (z < x1_sh[0]) {
          // left of first shell ---------------------------------
          rho = rho_amb + 0.5*corr_ampl; // adjustment by corr_ampl to ensure fair comparison
          press = press_amb * (rho/rho_amb); // keep the ambient temperature the same
          vel = vel_sh[0]; gamma = gamma_sh[0];
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
        } else if (z <= x2_sh[0]) {
          // inside first shell ---------------------------------
          rho = rho_sh[0]; press = press_sh[0]; vel = vel_sh[0]; gamma = gamma_sh[0];
          #ifdef MHD
          B = bfield_sh[0];
          #endif
          if (corr_type == 2) { // pressure perturbations inside shells
            if (corr_switch == 1) {
              press += corr_ampl * (press - press_amb)
                  * 0.5 * cos( 2.*M_PI *
                  (corr_nx * (z-x1_sh[0]) / (x2_sh[0]-x1_sh[0])
                 + corr_ny * (r-pDomain->MinX[1]) / (pDomain->MaxX[1]-pDomain->MinX[1]))
              - M_PI);
            } else { // ensure fair comparison
              ;
            }
          } else if (corr_type == 3) { // velocity perturbations inside shells
            if (corr_switch == 1) {
              // varying gamma^2 gives us conserved total energy
              gamma = sqrt(SQR(gamma) + corr_ampl * (SQR(gamma) - 1.0)
                  * 0.5 * cos( 2.*M_PI *
                  (corr_nx * (z-x1_sh[0]) / (x2_sh[0]-x1_sh[0])
                 + corr_ny * (r-pDomain->MinX[1]) / (pDomain->MaxX[1]-pDomain->MinX[1]))
              - M_PI));
              vel = (vel/fabs(vel)) * gamma2v(gamma);
            } else { // ensure fair comparison
              ;
            }
          }
        } else if (z < x1_sh[1]) {
          // between shells -------------------------------------
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
          if (corr_type == 1) { // density perturbations between shells
            if (corr_switch > 0) { // apply corrugation
              rho += (corr_ampl - rho_amb)
                  * 0.5 * ( cos( 2.*M_PI *
                       (corr_nx * (z-x2_sh[0]) / (x1_sh[1]-x2_sh[0])
                      + corr_ny * (r-pDomain->MinX[1]) / (pDomain->MaxX[1]-pDomain->MinX[1]))
                   - M_PI) +1 );
            } else {
              rho += 0.5*corr_ampl; // ensure fair comparison
            }
            press *= (rho/rho_amb); // keep the ambient temperature the same
          }
        } else if (z <= x2_sh[1]) {
          // inside second shell -----------------------------
          rho = rho_sh[1]; press = press_sh[1]; vel = vel_sh[1]; gamma = gamma_sh[1];
          #ifdef MHD
          B = bfield_sh[1];
          #endif
          if (corr_type == 2) { // pressure perturbations inside shells
            if (corr_switch == 1) {
              press += corr_ampl * (press - press_amb)
                  * 0.5 * cos( 2.*M_PI *
                  (corr_nx * (z-x1_sh[1]) / (x2_sh[1]-x1_sh[1])
                 + corr_ny * (r-pDomain->MinX[1]) / (pDomain->MaxX[1]-pDomain->MinX[1]))
              - M_PI);
            } else { // ensure fair comparison
              ;
            }
          } else if (corr_type == 3) { // velocity perturbations inside shells
            if (corr_switch == 1) {
              // varying gamma^2 gives us conserved total energy
              gamma = sqrt(SQR(gamma) + corr_ampl * (SQR(gamma) - 1.0)
                  * 0.5 * cos( 2.*M_PI *
                  (corr_nx * (z-x1_sh[1]) / (x2_sh[1]-x1_sh[1])
                 + corr_ny * (r-pDomain->MinX[1]) / (pDomain->MaxX[1]-pDomain->MinX[1]))
              - M_PI));
              vel = (vel/fabs(vel)) * gamma2v(gamma);
            } else { // ensure fair comparison
              ;
            }
          }
        } else {
          // right of second shell ---------------------------
          rho = rho_amb + 0.5*corr_ampl; // adjustment by corr_ampl to ensure fair comparison
          press = press_amb * (rho/rho_amb); // keep the ambient temperature the same
          vel = vel_sh[1]; gamma = gamma_sh[1];
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
        // NOTE: rho and Pg are in the fluid frame, see Beckwith & Stone 2011
        sqr_gamma = SQR(gamma);
        enthalpy = 1. + adiab_idx * press / ((adiab_idx-1.)*rho);
        pGrid->U[k][j][i].d = gamma*rho;
        pGrid->U[k][j][i].M1 = sqr_gamma*rho*enthalpy*vel;
        pGrid->U[k][j][i].M2 = 0.;
        pGrid->U[k][j][i].M3 = 0.;
        pGrid->U[k][j][i].E = sqr_gamma*rho*enthalpy - press;

        #ifdef MHD
        if (corr_type != 4 ||  corr_switch == 0) {
          // set bfield in the y-direction
          pGrid->U[k][j][i].B1c = 0.;
          pGrid->U[k][j][i].B2c = B;
          pGrid->U[k][j][i].B3c = 0.;
          if (i >= is) {
            pGrid->B2i[k][j][i] = B;
          }
        }

        // make Special-Relativity adjustments due to bfield
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
	if (corr_type != 4) {
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
	}
  #endif // MHD

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
