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
  Real x1min, x1max, L1, x2min, x2max, L2;
  x1min = pDomain->RootMinX[0];
  x1max = pDomain->RootMaxX[0];
  L1 = x1max - x1min;
  x2min = pDomain->RootMinX[1];
  x2max = pDomain->RootMaxX[1];
  L2 = x2max - x2min;

  // Read problem parameters
  Real rho = par_getd("problem", "rho");
  Real vel1 = par_getd("problem", "vel1");
  int npart = par_geti("particle", "parnumgrid");
  Real part_vel2 = par_getd("problem", "part_vel2");

  // Prepare the hydro grid
	for (k=ks; k<=ke; k++) {
	  for (j=js; j<=je; j++) {
	    for (i=is; i<=ie; i++) {
	      // resolve the physical location
        cc_pos(pGrid,i,j,k,&x1,&x2,&x3);
        // set hydro variables
        pGrid->U[k][j][i].d = rho;
        pGrid->U[k][j][i].M1 = rho * vel1;
        pGrid->U[k][j][i].M2 = 0.0;
        pGrid->U[k][j][i].M3 = 0.0;
        /*#ifndef BAROTROPIC
        pGrid->U[k][j][i].E = 2.5/Gamma_1
         + 0.5*(SQR(pGrid->U[k][j][i].M1) + SQR(pGrid->U[k][j][i].M2)
         + SQR(pGrid->U[k][j][i].M3))/pGrid->U[k][j][i].d;
        #endif // BAROTROPIC*/
	    }
	  }
	}

	// Prepare the particles
	pGrid->nparticle = 0; pgrid = 0;
	for (p = 0; p < npart; p++) {
	  // set particle location
	  pos.x1 = x1min + L1 * (0.5 + 1.0*p/(npart+1));
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
      pGrid->particle[pgrid].v1 = 0.;
      pGrid->particle[pgrid].v2 = part_vel2;
      pGrid->particle[pgrid].v3 = 0.;
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
