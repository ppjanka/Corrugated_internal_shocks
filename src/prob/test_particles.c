#include "copyright.h"
/*============================================================================*/
/*! \file test_particles.c
 *  \brief Problem for testing the particle module.
 *  -- based on par_epicycle.c, Bai & Stone
/*============================================================================*/

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "defs.h"
#include "athena.h"
#include "globals.h"
#include "prototypes.h"
#include "particles/particle.h"

// Special Relativity handling functions
inline Real v2gamma (Real vel)
{return 1./sqrt(1.-vel*vel);}
inline Real gamma2v (Real _gamma)
{return sqrt(1.-1./(_gamma*_gamma));}

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 *============================================================================*/

static bool part_in_rank (const Real3Vect pos);

/*------------------------ filewide global variables -------------------------*/
char name[50];

/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/
/* problem:  */

void problem(DomainS *pDomain)
{
  GridS *pGrid = pDomain->Grid;
  int i,j,k;
  int is,ie,js,je,ks,ke;
  is = pGrid->is; ie = pGrid->ie;
  js = pGrid->js; je = pGrid->je;
  ks = pGrid->ks; ke = pGrid->ke;
  Real x1, x2, x3;
  Real3Vect pos;
  long int p, pgrid;

  // Initialize boxsize
  Real x1min, x1max, L1, x2min, x2max, L2, x3min, x3max, L3;
  x1min = pDomain->RootMinX[0];
  x1max = pDomain->RootMaxX[0];
  L1 = x1max - x1min;
  x2min = pDomain->RootMinX[1];
  x2max = pDomain->RootMaxX[1];
  L2 = x2max - x2min;
  x3min = pDomain->RootMinX[2];
  x3max = pDomain->RootMaxX[2];
  L3 = x3max - x3min;

  // Read problem parameters
  #ifndef BAROTROPIC
  Real adiab_idx = par_getd("problem", "gamma");
  Real csound = par_getd("problem", "iso_csound");
  Real press;
  #endif //BAROTROPIC
  Real rho = par_getd("problem", "rho");
  Real vel1 = par_getd("problem", "vel1");
  Real vel2 = par_getd("problem", "vel2");
  Real vel3 = par_getd("problem", "vel3");
  int npart = par_geti("particle", "parnumgrid");
  Real part_vel1 = par_getd("problem", "part_vel1");
  Real part_vel2 = par_getd("problem", "part_vel2");
  Real part_vel3 = par_getd("problem", "part_vel3");
  #ifdef MHD
  Real bfield1 = par_getd("problem", "bfield1");
  int bfield1_type = par_geti("problem", "bfield1_type");
  Real bfield2 = par_getd("problem", "bfield2");
  int bfield2_type = par_geti("problem", "bfield2_type");
  Real bfield3 = par_getd("problem", "bfield3");
  int bfield3_type = par_geti("problem", "bfield3_type");
  #endif
  #ifdef SPECIAL_RELATIVITY
  Real enthalpy, gamma, sqr_b, sqr_gamma;
  #endif

  // Prepare the mhd grid
	for (k=ks; k<=ke; k++) {
	  for (j=js; j<=je; j++) {
      #pragma omp simd
	    for (i=is; i<=ie; i++) {
	      // resolve the physical location
        cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        #ifndef SPECIAL_RELATIVITY
        // set hydro variables
        pGrid->U[k][j][i].d = rho;
        pGrid->U[k][j][i].M1 = rho * vel1;// * cos(M_PI*(x1 - x1min)/(L1));
        pGrid->U[k][j][i].M2 = rho * vel2;
        pGrid->U[k][j][i].M3 = rho * vel3;
        // set magnetic fields
        #ifdef MHD
        pGrid->U[k][j][i].B1c = 0.0;
        pGrid->U[k][j][i].B2c = 0.0;
        pGrid->U[k][j][i].B3c = bfield3;// * sin(M_PI*(x1 - x1min)/(L1));
        #endif

        #ifndef BAROTROPIC
        press = SQR(csound) * rho / adiab_idx;
        pGrid->U[k][j][i].E = press / (adiab_idx - 1.0)
         + 0.5*(SQR(pGrid->U[k][j][i].M1) + SQR(pGrid->U[k][j][i].M2)
         + SQR(pGrid->U[k][j][i].M3))/pGrid->U[k][j][i].d;
        #ifdef MHD
        pGrid->U[k][j][i].E += 0.5 * (SQR(pGrid->U[k][j][i].B1c) + SQR(pGrid->U[k][j][i].B2c) + SQR(pGrid->U[k][j][i].B3c));
        #endif //MHD
        #endif // BAROTROPIC
        //---------------------------------------------------
        #else // SR case
        press = SQR(csound) * rho / adiab_idx;
        enthalpy = 1. + adiab_idx * press / ((adiab_idx-1.)*rho);
        gamma = v2gamma(sqrt(SQR(vel1) + SQR(vel2) + SQR(vel3)));
        // set hydro variables
        pGrid->U[k][j][i].d = gamma*rho;
        pGrid->U[k][j][i].M1 = gamma*gamma*rho*enthalpy*vel1;
        pGrid->U[k][j][i].M2 = gamma*gamma*rho*enthalpy*vel2;
        pGrid->U[k][j][i].M3 = gamma*gamma*rho*enthalpy*vel3;
        // set magnetic fields
        #ifdef MHD
        if (bfield1_type == 1) { // constant
          pGrid->U[k][j][i].B1c = bfield1;
        } else if (bfield1_type == 2) { // linear with x2
          pGrid->U[k][j][i].B1c = bfield1 * (0.5 + 0.5*(x2 - x2min)/L2);
        }
        if (bfield2_type == 1) { // constant
          pGrid->U[k][j][i].B2c = bfield2;
        } else if (bfield2_type == 2) { // linear with x3
          pGrid->U[k][j][i].B2c = bfield2 * (0.5 + 0.5*(x3 - x3min)/L3);
        }
        if (bfield3_type == 1) { // constant
          pGrid->U[k][j][i].B3c = bfield3;
        } else if (bfield3_type == 2) { // linear with x1
          pGrid->U[k][j][i].B3c = bfield3 * (0.5 + 0.5*(x1 - x1min)/L1);
        }
        // make momentum adjustments due to bfield
        sqr_gamma = SQR(gamma);
        sqr_b = SQR(pGrid->U[k][j][i].B1c) +
                 SQR(pGrid->U[k][j][i].B2c) +
                 SQR(pGrid->U[k][j][i].B3c);
        pGrid->U[k][j][i].M1 += sqr_b * vel1 /*from w_tot*/;
        pGrid->U[k][j][i].M2 += sqr_b * vel2 /*from w_tot*/;
        pGrid->U[k][j][i].M3 += sqr_b * vel3 /*from w_tot*/;
        #endif

        #ifndef BAROTROPIC
        pGrid->U[k][j][i].E = gamma*gamma*rho*enthalpy - press;
        #ifdef MHD
        // ignoring the following terms gets us near the pressure equillibrium
        //pGrid->U[k][j][i].E += sqr_b /*from w_tot*/;// - 0.5 * sqr_b / sqr_gamma /*from P_tot*/;
        #endif //MHD
        #endif // BAROTROPIC
        #endif // SPECIAL_RELATIVITY
	    }
	  }
	}

  #ifdef MHD
  // set the rest of face-centered bfield
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      #pragma omp simd
      for (i=is; i<=ie+1; i++) {
        // resolve the physical location
        fc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        if (bfield1_type == 1) { // constant
          pGrid->B1i[k][j][i] = bfield1;
        } else if (bfield1_type == 2) { // linear with x2
          pGrid->B1i[k][j][i] = bfield1 * (0.5 + 0.5*(x2 - x2min)/L2);
        }
      }
    }
  }
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je+1; j++) {
      #pragma omp simd
      for (i=is; i<=ie; i++) {
        // resolve the physical location
        fc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        if (bfield2_type == 1) { // constant
          pGrid->B2i[k][j][i] = bfield2;
        } else if (bfield2_type == 2) { // linear with x3
          pGrid->B2i[k][j][i] = bfield2 * (0.5 + 0.5*(x3 - x3min)/L3);
        }
      }
    }
  }
  for (k=ks; k<=(ke > 1 ? ke+1 : ke); k++) {
    for (j=js; j<=je; j++) {
      #pragma omp simd
      for (i=is; i<=ie; i++) {
        // resolve the physical location
        fc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        if (bfield3_type == 1) { // constant
          pGrid->B3i[k][j][i] = bfield3;
        } else if (bfield3_type == 2) { // linear with x3
          pGrid->B3i[k][j][i] = bfield3 * (0.5 + 0.5*(x1 - x1min)/L1);
        }
      }
    }
  }
  #endif //MHD

	// Prepare the particles
	tstop0[0] = par_getd_def("particle","tstop",1.0e20); // particle stopping time, sim.u.
  grproperty[0].alpha = par_getd("particle", "alpha"); /*!< charge-to-mass ratio, q/mc, see Mignone et al. (2018), eq. 18 */
	pGrid->nparticle = 0; pgrid = 0;
	for (p = 0; p < npart; p++) {
	  // set particle location
	  pos.x1 = x1min + L1 * ((0.5 + p)/(npart+1));
	  pos.x2 = 0.; pos.x3 = 0.;
	  if (part_in_rank(pos)) { // if in this MPI rank
	    (pGrid->nparticle)++;
	    if (pGrid->nparticle+2 > pGrid->arrsize)
	      particle_realloc(pGrid, pGrid->nparticle+2);
	    // particle properties
	    pGrid->particle[pgrid].property = 0;
      pGrid->particle[pgrid].x1 = pos.x1;
      pGrid->particle[pgrid].x2 = pos.x2;
      pGrid->particle[pgrid].x3 = pos.x3;
      pGrid->particle[pgrid].v1 = part_vel1;
      pGrid->particle[pgrid].v2 = part_vel2;
      pGrid->particle[pgrid].v3 = part_vel3;
      pGrid->particle[pgrid].pos = 1; /* grid particle */
      pGrid->particle[pgrid].my_id = p;
      #ifdef MPI_PARALLEL
      pGrid->particle[pgrid].init_id = myID_Comm_world;
      #endif
      pgrid++;
	  }
	}

	if (myID_Comm_world == 0) { // only on head rank
	  // format the output name
	  #ifdef MPI_PARALLEL
    sprintf(name, "../%s.dat","particleData");
	  #else
    sprintf(name, "%s.dat","particleData");
	  #endif
    // flush output file
    FILE *fid = fopen(name,"w");
    fclose(fid);
  }

	#ifdef MPI_PARALLEL
	  MPI_Bcast(name,50,MPI_CHAR,0,MPI_COMM_WORLD);
	#endif

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
  fwrite(name, sizeof(char),50,fp);
  return;
}

void problem_read_restart(MeshS *pM, FILE *fp)
{
  fread(name, sizeof(char),50,fp);
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
  return;
}

void Userwork_after_loop(MeshS *pM)
{
  return;
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
