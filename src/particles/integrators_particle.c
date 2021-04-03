#include "../copyright.h"
/*===========================================================================*/
/*! \file integrators_particle.c
 *  \brief Provide three kinds of particle integrators.
 *
 * PURPOSE: provide three kinds of particle integrators, namely, 2nd order
 *   explicit, 2nd order semi-implicit and 2nd order fully implicit.
 * 
 * CONTAINS PUBLIC FUNCTIONS:
 * - Integrate_Particles();
 * - int_par_exp   ()
 * - int_par_semimp()
 * - int_par_fulimp()
 * - feedback_predictor()
 * - feedback_corrector()
 *
 * PRIVATE FUNCTION PROTOTYPES:
 * - Delete_Ghost()   - delete ghost particles
 * - JudgeCrossing()  - judge if the particle cross the grid boundary
 * - Get_Drag()       - calculate the drag force
 * - Get_Force()      - calculate forces other than the drag
 *
 * REFERENCE:
 *   X.-N. Bai & J.M. Stone, 2010, ApJS, 190, 297 									      */
/*============================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../defs.h"
#include "../athena.h"
#include "../prototypes.h"
#include "prototypes.h"
#include "particle.h"
#include "../globals.h"

#ifdef PARTICLES         /* endif at the end of the file */

/*==============================================================================
 * PRIVATE FUNCTION PROTOTYPES:
 *   Delete_Ghost()   - delete ghost particles
 *   JudgeCrossing()  - judge if the particle cross the grid boundary
 *   Get_Drag()       - calculate the drag force
 *   Get_Force()      - calculate forces other than the drag
 *   Get_ForceDiff()  - calculate the force difference between particle and gas
 *============================================================================*/
void   Delete_Ghost(GridS *pG);
void   JudgeCrossing(GridS *pG, Real x1, Real x2, Real x3, GrainS *gr);
Real3Vect Get_Drag(GridS *pG, int type, Real x1, Real x2, Real x3,
                Real v1, Real v2, Real v3, Real3Vect cell1, Real *tstop1);
Real3Vect Get_Force(GridS *pG, Real x1, Real x2, Real x3,
                               Real v1, Real v2, Real v3);

/*=========================== PUBLIC FUNCTIONS ===============================*/
/*----------------------------------------------------------------------------*/

/*---------------------------------- Main Integrator -------------------------*/
/*! \fn void Integrate_Particles(DomainS *pD)
 *  \brief Main particle integrator.
 *
 * Input: Grid which is already evolved in half a time step. Paricles unevolved.
 * Output: particle updated for one full time step; feedback array for corrector
 *         updated.
 * Note: This routine allows has the flexibility to freely choose the particle
 *       integrator.
 * Should use fully implicit integrator for tightly coupoled particles.
 * Otherwise the semi-implicit integrator performs better.
 */
void Integrate_Particles(DomainS *pD)
{
  GrainS *curG, *curP, mygr;    /* pointer of the current working position */
  long p;                       /* particle index */
  Real dv1, dv2, dv3, ts, t1;   /* amount of velocity update, stopping time */
  Real3Vect cell1;              /* one over dx1, dx2, dx3 */

  GridS *pG = pD->Grid;         /* set ptr to Grid */

/* Initialization */
#ifdef FEEDBACK
  feedback_clear(pG);   /* clean the feedback array */
#endif /* FEEDBACK */

  curP = &(mygr);       /* temperory particle */

  /* cell1 is a shortcut expressions as well as dimension indicator */
  if (pG->Nx[0] > 1)  cell1.x1 = 1.0/pG->dx1;  else cell1.x1 = 0.0;
  if (pG->Nx[1] > 1)  cell1.x2 = 1.0/pG->dx2;  else cell1.x2 = 0.0;
  if (pG->Nx[2] > 1)  cell1.x3 = 1.0/pG->dx3;  else cell1.x3 = 0.0;

  /* delete all ghost particles */
  Delete_Ghost(pG);

  p = 0;
  while (p<pG->nparticle)
  {/* loop over all particles */
    curG = &(pG->particle[p]);

    printf("INITIAL STATE: x = v = %.2e %.2e %.2e, v = %.2e %.2e %.2e\n", curG->x1, curG->x2, curG->x3, curG->v1, curG->v2, curG->v3);
    //exit(0);

/* Step 1: Calculate velocity update */
    switch(grproperty[curG->property].integrator)
    {
      case 1: /* 2nd order explicit integrator */
        int_par_exp(pG, curG, cell1, &dv1, &dv2, &dv3, &ts);
        break;

      case 2: /* 2nd order semi-implicit integrator */
        int_par_semimp(pG, curG, cell1, &dv1, &dv2, &dv3, &ts);
        break;

      case 3: /* 2nd order fully implicit integrator */
        int_par_fulimp(pG, curG, cell1, &dv1, &dv2, &dv3, &ts);
        break;

      case 4: /* Boris pusher, see Mignone et al. (2018) */
        #if defined(SPECIAL_RELATIVITY) && defined(VL_INTEGRATOR) && defined(CARTESIAN)
        int_par_boris(pG, curG, cell1, &dv1, &dv2, &dv3, &ts);
        #else
        ath_error("[integrate_particle]: the Boris integrator has only been tested for the VL-SR integrator in Cartesian coords!");
        #endif
        break;

      default:
        ath_error("[integrate_particle]: unknown integrator type!");
    }

/* Step 2: particle update to curP */

    /* velocity update */
    curP->v1 = curG->v1 + dv1;
    curP->v2 = curG->v2 + dv2;
    curP->v3 = curG->v3 + dv3;

    /* position update */
    if (pG->Nx[0] > 1)
      curP->x1 = curG->x1 + 0.5*pG->dt*(curG->v1 + curP->v1);
    else /* do not move if this dimension collapses */
      curP->x1 = curG->x1;

    if (pG->Nx[1] > 1)
      curP->x2 = curG->x2 + 0.5*pG->dt*(curG->v2 + curP->v2);
    else /* do not move if this dimension collapses */
      curP->x2 = curG->x2;

    if (pG->Nx[2] > 1)
      curP->x3 = curG->x3 + 0.5*pG->dt*(curG->v3 + curP->v3);
    else /* do not move if this dimension collapses */
      curP->x3 = curG->x3;

#ifdef FARGO
    /* shift = -qshear * Omega_0 * x * dt */
    pG->parsub[p].shift = -0.5*qshear*Omega_0*(curG->x1+curP->x1)*pG->dt;
#endif

/* Step 3: calculate feedback force to the gas */
#ifdef FEEDBACK
    feedback_corrector(pG, curG, curP, cell1, dv1, dv2, dv3, ts);
#endif /* FEEDBACK */

/* Step 4: Final update of the particle */
    /* update particle status (crossing boundary or not) */
    JudgeCrossing(pG, curP->x1, curP->x2, curP->x3, curG);

    /* update the particle */
    curG->x1 = curP->x1;
    curG->x2 = curP->x2;
    curG->x3 = curP->x3;
    curG->v1 = curP->v1;
    curG->v2 = curP->v2;
    curG->v3 = curP->v3;
    p++;

    printf("FINAL STATE: x = v = %.2e %.2e %.2e, v = %.2e %.2e %.2e\n", curG->x1, curG->x2, curG->x3, curG->v1, curG->v2, curG->v3);

  } /* end of the for loop */

  /* output the status */
  ath_pout(0, "In processor %d, there are %ld particles.\n",
                           myID_Comm_world, pG->nparticle);

  return;
}

/*! \fn inline void cross_product (Real* vec1, Real* vec2, Real* out)
 *  \brief cross product of two vectors
 */
inline void cross_product (Real* vec1, Real* vec2, Real* out)
{
  for (int n = 0; n < 3; n++)
    out[n] = vec1[(n+1)%3]*vec2[(n+2)%3] - vec1[(n+2)%3]*vec2[(n+1)%3];
}
/*! \fn inline void velocity_3to4 (Real* v, Rreal* u, Real* gamma)
 *  \brief a function to transform from 3-velocity to 4-velocity
 */
inline void velocity_3to4 (Real* v, Real* u, Real* gamma)
{
  (*gamma) = 1.0 / sqrt(1.0 - SQR(v[0]) - SQR(v[1]) - SQR(v[2]));
  for (int n = 0; n < 3; n++)
    u[n] = (*gamma) * v[n];
}
/*! \fn inline void velocity_4to3 (Real* u, Real* v, Real* gamma)
 *  \brief a function to transform from 4-velocity to 3-velocity
 */
inline void velocity_4to3 (Real* u, Real* v, Real* gamma)
{
  (*gamma) = sqrt(1.0 + SQR(u[0]) + SQR(u[1]) + SQR(u[2]));
  for (int n = 0; n < 3; n++)
    v[n] = u[n] / (*gamma);
}
/*! \fn void interpolate_EM (GridS *pG, Real x1, Real x2, Real x3,
    Real* B, Real* E, Real* v)
 *  \brief interpolate B and E fields at the current particle position
 *   -- uses Triangular Shape Cloud (TSC) for interpolation, see Mignone et al. (2018), eq. 44
 *   -- B and E are assumed to be 3-element arrays
 */
void interpolate_EM (GridS *pG, Real x1, Real x2, Real x3,
    Real* B, Real* E, Real* v, Real3Vect cell1)
{
  // TODO: find a way to get rid of all those if statements...

  // find the indices of the nearest cell center (i+1,j+1,k+1)
  // calculate interpolation weights for that location
  int i,j,k;
  Real x,y,z; int ii,jj,kk,nn, idxk, idxj, idxi;
  Real weights[3][3][3];
  /*if (pG->Nx[0] > 1) {
    i = (int) round((x1-pG->MinX[0])/(pG->dx1) - 0.5) + pG->is;
  } else {
    i = 1;
  }
  if (pG->Nx[1] > 1) {
    j = (int) round((x2-pG->MinX[1])/(pG->dx2) - 0.5) + pG->js;
  } else {
    j = 1;
  }
  if (pG->Nx[2] > 1) {
    k = (int) round((x3-pG->MinX[2])/(pG->dx3) - 0.5) + pG->ks;
  } else {
    k = 1;
  }*/
  printf("Getting weights\n");
  getweight(pG, x,y,z, cell1, weights, &i,&j,&k);
  printf("done\n");

  // interpolate bfields and vels from cell-centered values
  // (which should have been evolved by dt/2 at this point)
  Real dist, weight, sum_weights = 0.0;
  #pragma omp simd
  for(nn = 0; nn < 3; nn++) {
    v[nn] = 0.0;
    B[nn] = 0.0;
  }
  //printf("velocity cleared.\n");
  for (kk = 0; kk < ( pG->Nx[2] > 1 ? 3 : 1 ); kk++) {
    for (jj = 0; jj < ( pG->Nx[1] > 1 ? 3 : 1 ); jj++) {
      #pragma omp simd
      for (ii = 0; ii < ( pG->Nx[0] > 1 ? 3 : 1 ); ii++) {
        printf(" -- velocity interp loop: %i %i %i\n", ii,jj,kk);
        idxk = k+kk;
        idxj = j+jj;
        idxi = i+ii;
        cc_pos(pG,idxi,idxj,idxk,&x,&y,&z);
        printf("1\n");
        // Triangular Shaped Cloud (TSC) weighing:
        weight = weights[kk][jj][ii];
        /*1.0;
        // x1 weight
        dist = fabs(x1-x)/pG->dx1;
        if (ii == 1) { // center point
          weight *= 0.75 - SQR(dist);
        } else {
          weight *= 0.5 * SQR(0.5 - dist);
        }
        // x2 weight
        dist = fabs(x2-y)/pG->dx2;
        if (jj == 1) { // center point
          weight *= 0.75 - SQR(dist);
        } else {
          weight *= 0.5 * SQR(0.5 - dist);
        }
        // x3 weight
        dist = fabs(x3-z)/pG->dx3;
        if (kk == 1) { // center point
          weight *= 0.75 - SQR(dist);
        } else {
          weight *= 0.5 * SQR(0.5 - dist);
        }*/
        printf("2\n");
        // maybe better to use conserved vars at half-step, [TODO:] move to SR (with VL-SR integrator it may be pre-computed)
        //printf("%i %i %i\n", idxk, idxj, idxi);
        v[0] += weight * pG->Whalf[idxk][idxj][idxi].V1;
        v[1] += weight * pG->Whalf[idxk][idxj][idxi].V2;
        v[2] += weight * pG->Whalf[idxk][idxj][idxi].V3;
        B[0] += weight * pG->Whalf[idxk][idxj][idxi].B1c;
        B[1] += weight * pG->Whalf[idxk][idxj][idxi].B2c;
        B[2] += weight * pG->Whalf[idxk][idxj][idxi].B3c;
        printf("weight: %.2e\n", weight);
        printf("v: %.2e %.2e %.2e\n", v[0], v[1], v[2]);
        printf("B: %.2e %.2e %.2e\n", B[0], B[1], B[2]);
        sum_weights += weight;
        //printf("3\n");
      }
    }
  }
  // finalize
  #pragma omp simd
  for(nn = 0; nn < 3; nn++) {
    v[nn] /= sum_weights;
    B[nn] /= sum_weights;
  }
  printf("Final v: %.2e %.2e %.2e\n", v[0], v[1], v[2]);
  printf("Final B: %.2e %.2e %.2e\n", B[0], B[1], B[2]);

  //printf("Velocity interpolated.\n");

  // CALCULATE E = - V x B
  cross_product(B, v, E);

  //printf("EM interpolated.\n");
}

/* ------------ Boris pusher -- implicit Verlet particle integrator -----------------*/
/*! \fn void int_par_boris(Grid *pG, Grain *curG, Real3Vect cell1,
 *                            Real *dv1, Real *dv2, Real *dv3, Real *ts)
 *  \brief Boris pusher -- implicit Verlet particle integrator (see Mignone et al. (2018)
 *  NOTE: only includes Lorentz force
 *  NOTE: Designed and tested only for the VL-SR integrator
 *
 *  TODO: Add a time-step constraint (possibly through particle multi-timestepping?)
 *
 * Input:
 *   grid pointer (pG), grain pointer (curG), cell size indicator (cell1)
 * Output:
 *   dv1,dv2,dv3: velocity update
 */
void int_par_boris(GridS *pG, GrainS *curG, Real3Vect cell1,
                              Real *dv1, Real *dv2, Real *dv3, Real *ts)
{
  printf("Inside Boris integrator.\n");

  int n;

  // SR quantities
  Real gamma, gamma_n; // particle and half-step fluid Lorentz factor
  Real u[3]; // particle 4-velocity
  Real v[3], vn[3]; // particle and half-step fluid 3-velocities
  v[0] = curG->v1; v[1] = curG->v2; v[2] = curG->v3;
  velocity_3to4(v, u, &gamma);
  printf("SR quantities calculated.\n");
  printf("v: %.2e %.2e %.2e\n", v[0], v[1], v[2]);
  printf("u: %.2e %.2e %.2e\n", u[0], u[1], u[2]);

  // integration step from the particle's charge-to-mass ratio, see Mignone et al. (2018), eq. 18
  Real h2 = 0.5 * grproperty[curG->property].alpha* pG->dt;

  // Step 1 [DRIFT]: predict the particle position after half time step
  printf("Step 1: ");
  Real x1n, x2n, x3n;
  if (pG->Nx[0] > 1)  x1n = curG->x1 + 0.5*curG->v1*pG->dt;
  else x1n = curG->x1;
  if (pG->Nx[1] > 1)  x2n = curG->x2 + 0.5*curG->v2*pG->dt;
  else x2n = curG->x2;
  if (pG->Nx[2] > 1)  x3n = curG->x3 + 0.5*curG->v3*pG->dt;
  else x3n = curG->x3;
  printf("done.\n");

  // Interpolate and extract electromagnetic components
  printf("EM components: \n");
  Real Bn[3], En[3]; // EM fields interpolated at the half-step particle position
  interpolate_EM(pG, x1n, x2n, x3n, Bn, En, vn, cell1);
  gamma_n = 1.0 / sqrt(1.0 - SQR(vn[0]) - SQR(vn[1]) - SQR(vn[2]));
  for (n = 0; n < 3; n++) {
    Bn[n] *= h2/gamma_n;
  }
  Real sqr_Bn = SQR(Bn[0]) + SQR(Bn[1]) + SQR(Bn[2]);
  printf("Bn: %.2e %.2e %.2e\n", Bn[0], Bn[1], Bn[2]);
  printf("En: %.2e %.2e %.2e\n", En[0], En[1], En[2]);
  printf("done.\n");

  // Step 2 [KICK]
  printf("Step 2: ");
  printf("u: %.2e %.2e %.2e\n", u[0], u[1], u[2]);
  Real u_minus [3];
  for (n = 0; n < 3; n++) {
    u_minus[n] = u[n] + h2 /*c*/ * En[n];
  }
  printf("u_minus: %.2e %.2e %.2e\n", u_minus[0], u_minus[1], u_minus[2]);
  printf("done.\n");

  // Step 3 [ROTATE]
  printf("Step 3: ");
  Real u_plus [3], buffer [3];
  cross_product(u_minus, Bn, buffer);
  printf("buffer: %.2e %.2e %.2e\n", buffer[0], buffer[1], buffer[2]);
  for (n = 0; n < 3; n++) {
    buffer[n] += u_minus[n];
    buffer[n] /= 0.5 * (1.0 + sqr_Bn);
  }
  printf("buffer: %.2e %.2e %.2e\n", buffer[0], buffer[1], buffer[2]);
  cross_product(buffer, Bn, u_plus);
  printf("u_plus: %.2e %.2e %.2e\n", u_plus[0], u_plus[1], u_plus[2]);
  for (n = 0; n < 3; n++) {
    u_plus[n] += u_minus[n];
  }
  printf("u_plus: %.2e %.2e %.2e\n", u_plus[0], u_plus[1], u_plus[2]);
  printf("done.\n");

  // Step 4 [KICK]
  printf("Step 4: ");
  for (n = 0; n < 3; n++) {
    u[n] = u_plus[n] + h2 /* c */ * En[n];
  }
  printf("New 4velocity: %.2e %.2e %.2e\n", u[0], u[1], u[2]);
  velocity_4to3(u, v, &gamma);
  printf("done.\n");

  // Report velocity change to the main integrator
  printf("Reporting velocity change: \n");
  printf("New velocity: %.2e %.2e %.2e\n", v[0], v[1], v[2]);
  (*dv1) = v[0] - curG->v1;
  (*dv2) = v[1] - curG->v2;
  (*dv3) = v[2] - curG->v3;
  printf(" -- velocity change: %.2e %.2e %.2e\n", (*dv1), (*dv2), (*dv3));
  printf("done.\n");

  // Step 5 [DRIFT] -- performed in Integrate_Particles (see above)
  printf("Boris integrator done.\n");

}

/* ------------ 2nd order fully implicit particle integrator -----------------*/
/*! \fn void int_par_fulimp(Grid *pG, Grain *curG, Real3Vect cell1, 
 *                            Real *dv1, Real *dv2, Real *dv3, Real *ts)
 *  \brief 2nd order fully implicit particle integrator
 *
 * Input: 
 *   grid pointer (pG), grain pointer (curG), cell size indicator (cell1)
 * Output:
 *   dv1,dv2,dv3: velocity update
 */
void int_par_fulimp(GridS *pG, GrainS *curG, Real3Vect cell1, 
                              Real *dv1, Real *dv2, Real *dv3, Real *ts)
{
  Real x1n, x2n, x3n;	/* first order new position at half a time step */
  Real3Vect fd, fr;	/* drag force and other forces */
  Real3Vect fc, fp, ft;	/* force at current & predicted position, total force */
  Real ts11, ts12;	/* 1/stopping time */
  Real b0,A,B,C,D,Det1;	/* matrix elements and determinant */
#ifdef SHEARING_BOX
  Real oh, oh2;		/* Omega_0*dt and its square */
#endif

/* step 1: predict of the particle position after one time step */
  if (pG->Nx[0] > 1)  x1n = curG->x1+curG->v1*pG->dt;
  else x1n = curG->x1;
  if (pG->Nx[1] > 1)  x2n = curG->x2+curG->v2*pG->dt;
  else x2n = curG->x2;
  if (pG->Nx[2] > 1)  x3n = curG->x3+curG->v3*pG->dt;
  else x3n = curG->x3;

#ifdef SHEARING_BOX
#ifndef FARGO
  /* advection part */
  if (ShBoxCoord == xy) x2n -= 0.5*qshear*curG->v1*SQR(pG->dt);
#endif
#endif

/* step 2: calculate the force at current position */
  fd = Get_Drag(pG, curG->property, curG->x1, curG->x2, curG->x3,
                                    curG->v1, curG->v2, curG->v3, cell1, &ts11);

  fr = Get_Force(pG, curG->x1, curG->x2, curG->x3,
                     curG->v1, curG->v2, curG->v3);

  fc.x1 = fd.x1+fr.x1;
  fc.x2 = fd.x2+fr.x2;
  fc.x3 = fd.x3+fr.x3;

/* step 3: calculate the force at the predicted positoin */
  fd = Get_Drag(pG, curG->property, x1n, x2n, x3n,
                                    curG->v1, curG->v2, curG->v3, cell1, &ts12);

  fr = Get_Force(pG, x1n, x2n, x3n, curG->v1, curG->v2, curG->v3);

  fp.x1 = fd.x1+fr.x1;
  fp.x2 = fd.x2+fr.x2;
  fp.x3 = fd.x3+fr.x3;

/* step 4: calculate the velocity update */
  /* shortcut expressions */
  b0 = 1.0+pG->dt*ts11;

  /* Total force */
  ft.x1 = 0.5*(fc.x1+b0*fp.x1);
  ft.x2 = 0.5*(fc.x2+b0*fp.x2);
  ft.x3 = 0.5*(fc.x3+b0*fp.x3);

#ifdef SHEARING_BOX
  oh = Omega_0*pG->dt;
  if (ShBoxCoord == xy) {/* (x1,x2,x3)=(X,Y,Z) */
    ft.x1 += -oh*fp.x2;
  #ifdef FARGO
    ft.x2 += 0.5*(2.0-qshear)*oh*fp.x1;
  #else
    ft.x2 += oh*fp.x1;
  #endif
  } else {               /* (x1,x2,x3)=(X,Z,Y) */
    ft.x1 += -oh*fp.x3;
  #ifdef FARGO
    ft.x3 += 0.5*(2.0-qshear)*oh*fp.x1;
  #else
    ft.x3 += oh*fp.x1;
  #endif
  }
#endif /* SHEARING_BOX */

  /* calculate the inverse matrix elements */
  D = 1.0+0.5*pG->dt*(ts11 + ts12 + pG->dt*ts11*ts12);
#ifdef SHEARING_BOX
  oh2 = SQR(oh);
  B = oh * (-2.0-(ts11+ts12)*pG->dt);
#ifdef FARGO
  A = D - (2.0-qshear)*oh2;
  C = 0.5*(qshear-2.0)*B;
#else /* FARGO */
  A = D - 2.0*oh2;
  C = -B;
#endif /* FARGO */
  Det1 = 1.0/(SQR(A)-B*C);
  if (ShBoxCoord == xy) {
    *dv1 = pG->dt*Det1*(ft.x1*A-ft.x2*B);
    *dv2 = pG->dt*Det1*(-ft.x1*C+ft.x2*A);
    *dv3 = pG->dt*ft.x3/D;
  } else {
    *dv1 = pG->dt*Det1*(ft.x1*A-ft.x3*B);
    *dv3 = pG->dt*Det1*(-ft.x1*C+ft.x3*A);
    *dv2 = pG->dt*ft.x2/D;
  }
#else /* SHEARING_BOX */
  D = 1.0/D;
  *dv1 = pG->dt*ft.x1*D;
  *dv2 = pG->dt*ft.x2*D;
  *dv3 = pG->dt*ft.x3*D;
#endif /* SHEARING_BOX */

  *ts = 0.5/ts11+0.5/ts12;

  return;
}


/*--------------- 2nd order semi-implicit particle integrator ----------------*/
/*! \fn void int_par_semimp(Grid *pG, Grain *curG, Real3Vect cell1, 
 *                            Real *dv1, Real *dv2, Real *dv3, Real *ts)
 *  \brief 2nd order semi-implicit particle integrator 
 *
 * Input: 
 *   grid pointer (pG), grain pointer (curG), cell size indicator (cell1)
 * Output:
 *   dv1,dv2,dv3: velocity update
 */
void int_par_semimp(GridS *pG, GrainS *curG, Real3Vect cell1, 
                              Real *dv1, Real *dv2, Real *dv3, Real *ts)
{
  Real3Vect fd, fr, ft;	/* drag force and other forces, total force */
  Real ts1, b, b2;	/* other shortcut expressions */
  Real x1n, x2n, x3n;	/* first order new position at half a time step */
#ifdef SHEARING_BOX
  Real b1, oh;		/* Omega_0*h */
#endif

/* step 1: predict of the particle position after half a time step */
  if (pG->Nx[0] > 1)  x1n = curG->x1+0.5*curG->v1*pG->dt;
  else x1n = curG->x1;
  if (pG->Nx[1] > 1)  x2n = curG->x2+0.5*curG->v2*pG->dt;
  else x2n = curG->x2;
  if (pG->Nx[2] > 1)  x3n = curG->x3+0.5*curG->v3*pG->dt;
  else x3n = curG->x3;

#ifdef SHEARING_BOX
#ifndef FARGO
  /* advection part */
  if (ShBoxCoord == xy) x2n -= 0.125*qshear*curG->v1*SQR(pG->dt);
#endif
#endif

/* Step 2: interpolation to get fluid density, velocity and the sound speed at\  * predicted position
 */
  fd = Get_Drag(pG, curG->property, x1n, x2n, x3n,
                                    curG->v1, curG->v2, curG->v3, cell1, &ts1);

  fr = Get_Force(pG, x1n, x2n, x3n, curG->v1, curG->v2, curG->v3);

  ft.x1 = fd.x1+fr.x1;
  ft.x2 = fd.x2+fr.x2;
  ft.x3 = fd.x3+fr.x3;

/* step 3: calculate velocity update */

  /* shortcut expressions */
  b = pG->dt*ts1+2.0;
#ifdef SHEARING_BOX
  oh = Omega_0*pG->dt;
#ifdef FARGO
  b1 = 1.0/(SQR(b)+2.0*(2.0-qshear)*SQR(oh));
#else
  b1 = 1.0/(SQR(b)+4.0*SQR(oh));
#endif /* FARGO */
  b2 = b*b1;
#else
  b2 = 1.0/b;
#endif /* SHEARING BOX */

    /* velocity evolution */
#ifdef SHEARING_BOX
  if (ShBoxCoord == xy)
  {/* (x1,x2,x3)=(X,Y,Z) */
    *dv1 = pG->dt*2.0*b2*ft.x1 + pG->dt*4.0*oh*b1*ft.x2;
  #ifdef FARGO
    *dv2 = pG->dt*2.0*b2*ft.x2 - 2.0*(2.0-qshear)*pG->dt*oh*b1*ft.x1;
  #else
    *dv2 = pG->dt*2.0*b2*ft.x2 - 4.0*pG->dt*oh*b1*ft.x1;
  #endif /* FARGO */
    *dv3 = pG->dt*2.0*ft.x3/b;
  }
  else
  {/* (x1,x2,x3)=(X,Z,Y) */
    *dv1 = pG->dt*2.0*b2*ft.x1 + pG->dt*4.0*oh*b1*ft.x3;
    *dv2 = pG->dt*2.0*ft.x2/b;
  #ifdef FARGO
    *dv3 = pG->dt*2.0*b2*ft.x3 - 2.0*(2.0-qshear)*pG->dt*oh*b1*ft.x1;
  #else
    *dv3 = pG->dt*2.0*b2*ft.x3 - 4.0*pG->dt*oh*b1*ft.x1;
  #endif
  }
#else
  *dv1 = pG->dt*2.0*b2*ft.x1;
  *dv2 = pG->dt*2.0*b2*ft.x2;
  *dv3 = pG->dt*2.0*b2*ft.x3;
#endif /* SHEARING_BOX */

  *ts = 1.0/ts1;

  return;
}


/*------------------- 2nd order explicit particle integrator -----------------*/
/*! \fn void int_par_exp(Grid *pG, Grain *curG, Real3Vect cell1,
 *                         Real *dv1, Real *dv2, Real *dv3, Real *ts)
 *  \brief 2nd order explicit particle integrator 
 *
 * Input: 
 *   grid pointer (pG), grain pointer (curG), cell size indicator (cell1)
 * Output:
 *   dv1,dv2,dv3: velocity update
 */
void int_par_exp(GridS *pG, GrainS *curG, Real3Vect cell1,
                           Real *dv1, Real *dv2, Real *dv3, Real *ts)
{
  Real3Vect fd, fr, ft;	/* drag force and other forces, total force */
  Real ts1;		/* 1/stopping time */
  Real x1n, x2n, x3n;	/* first order new position at half a time step */
  Real v1n, v2n, v3n;	/* first order new velocity at half a time step */

/* step 1: predict of the particle position after half a time step */
  if (pG->Nx[0] > 1)
    x1n = curG->x1+0.5*curG->v1*pG->dt;
  else x1n = curG->x1;
  if (pG->Nx[1] > 1)
    x2n = curG->x2+0.5*curG->v2*pG->dt;
  else x2n = curG->x2;
  if (pG->Nx[2] > 1)
    x3n = curG->x3+0.5*curG->v3*pG->dt;
  else x3n = curG->x3;

#ifdef SHEARING_BOX
#ifndef FARGO
  /* advection part */
  if (ShBoxCoord == xy) x2n -= 0.125*qshear*curG->v1*SQR(pG->dt);
#endif
#endif

/* step 2: calculate the force at current position */
  fd = Get_Drag(pG, curG->property, curG->x1, curG->x2, curG->x3,
                                    curG->v1, curG->v2, curG->v3, cell1, &ts1);

  fr = Get_Force(pG, curG->x1, curG->x2, curG->x3,
                     curG->v1, curG->v2, curG->v3);

  ft.x1 = fd.x1+fr.x1;
  ft.x2 = fd.x2+fr.x2;
  ft.x3 = fd.x3+fr.x3;

  v1n = curG->v1 + 0.5*ft.x1*pG->dt;
  v2n = curG->v2 + 0.5*ft.x2*pG->dt;
  v3n = curG->v3 + 0.5*ft.x3*pG->dt;

/* step 3: calculate the force at the predicted positoin */
  fd = Get_Drag(pG, curG->property, x1n, x2n, x3n, v1n, v2n, v3n, cell1, &ts1);

  fr = Get_Force(pG, x1n, x2n, x3n, v1n, v2n, v3n);

  ft.x1 = fd.x1+fr.x1;
  ft.x2 = fd.x2+fr.x2;
  ft.x3 = fd.x3+fr.x3;

/* step 4: calculate velocity update */
  *dv1 = ft.x1*pG->dt;
  *dv2 = ft.x2*pG->dt;
  *dv3 = ft.x3*pG->dt;

  *ts = 1.0/ts1;

  return;
}

#ifdef FEEDBACK

/*! \fn void feedback_predictor(GridS *pG)
 *  \brief Calculate the feedback of the drag force from the particle to the gas
 *
 * Serves for the predictor step. It deals with all the particles.
 * Input: pG: grid with particles
 * Output: pG: the array of drag forces exerted by the particle is updated
*/
void feedback_predictor(DomainS *pD)
{
  GridS *pG = pD->Grid;
  int is,js,ks,i,j,k;
  long p;                   /* particle index */
  Real weight[3][3][3];     /* weight function */
  Real rho, cs, tstop;      /* density, sound speed, stopping time */
  Real u1, u2, u3;
  Real vd1, vd2, vd3, vd;   /* velocity difference between particle and gas */
  Real f1, f2, f3;          /* feedback force */
  Real m, ts1h;             /* grain mass, 0.5*dt/tstop */
  Real3Vect cell1;          /* one over dx1, dx2, dx3 */
  Real3Vect fb;             /* drag force, fluid velocity */
#ifndef BAROTROPIC
  Real Elosspar;            /* energy dissipation rate due to drag */
#endif
  Real stiffness;           /* stiffness parameter of feedback */
  GrainS *gr;              /* pointer of the current working position */

  /* initialization */
  get_gasinfo(pG);          /* calculate gas information */

  for (k=klp; k<=kup; k++)
    for (j=jlp; j<=jup; j++)
      for (i=ilp; i<=iup; i++) {
        /* clean the feedback array */
        pG->Coup[k][j][i].fb1 = 0.0;
        pG->Coup[k][j][i].fb2 = 0.0;
        pG->Coup[k][j][i].fb3 = 0.0;
#ifndef BAROTROPIC
        pG->Coup[k][j][i].Eloss = 0.0;
#endif
        pG->Coup[k][j][i].FBstiff = 0.0;
      }

  /* convenient expressions */
  if (pG->Nx[0] > 1)  cell1.x1 = 1.0/pG->dx1;
  else                cell1.x1 = 0.0;

  if (pG->Nx[1] > 1)  cell1.x2 = 1.0/pG->dx2;
  else                cell1.x2 = 0.0;

  if (pG->Nx[2] > 1)  cell1.x3 = 1.0/pG->dx3;
  else                cell1.x3 = 0.0;

  /* loop over all particles to calculate the drag force */
  for (p=0; p<pG->nparticle; p++)
  {/* loop over all particle */
    gr = &(pG->particle[p]);

    /* interpolation to get fluid density and velocity */
    getweight(pG, gr->x1, gr->x2, gr->x3, cell1, weight, &is, &js, &ks);
    if (getvalues(pG, weight, is, js, ks,
                              &rho, &u1, &u2, &u3, &cs, &stiffness) == 0)
    { /* particle is in the grid */

      /* apply gas velocity shift due to pressure gradient */
      gasvshift(gr->x1, gr->x2, gr->x3, &u1, &u2, &u3);
      /* velocity difference */
      vd1 = u1 - gr->v1;
      vd2 = u2 - gr->v2;
      vd3 = u3 - gr->v3;
      vd = sqrt(vd1*vd1 + vd2*vd2 + vd3*vd3);

      /* calculate particle stopping time */
      tstop = get_ts(pG, gr->property, rho, cs, vd);
      tstop = MAX(tstop, pG->dt);
      ts1h = 0.5*pG->dt/tstop;

      /* Drag force density */
      m = grproperty[gr->property].m;
      fb.x1 = m * vd1 * ts1h;
      fb.x2 = m * vd2 * ts1h;
      fb.x3 = m * vd3 * ts1h;

      /* calculate feedback stiffness */
      stiffness = m*pG->dt/tstop;

      /* distribute the drag force (density) to the grid */
#ifndef BAROTROPIC
      Elosspar = fb.x1*vd1 + fb.x2*vd2 + fb.x3*vd3;
      distrFB_pred(pG, weight, is, js, ks, fb, stiffness, Elosspar);
#else
      distrFB_pred(pG, weight, is, js, ks, fb, stiffness);
#endif
    }
  }/* end of the for loop */

/* normalize stiffness and correct for feedback */
  for (k=klp; k<=kup; k++)
    for (j=jlp; j<=jup; j++)
      for (i=ilp; i<=iup; i++)
      {
        pG->Coup[k][j][i].FBstiff /= pG->U[k][j][i].d;

        stiffness = 1.0/MAX(1.0, pG->Coup[k][j][i].FBstiff);

        pG->Coup[k][j][i].fb1 *= stiffness;
        pG->Coup[k][j][i].fb2 *= stiffness;
        pG->Coup[k][j][i].fb3 *= stiffness;
      }

  return;
}

/*----------------------------------------------------------------------------*/
/*! \fn void feedback_corrector(GridS *pG, GrainS *gri, GrainS *grf,
 *                    Real3Vect cell1, Real dv1, Real dv2, Real dv3, Real ts)
 *  \brief  Calculate the feedback of the drag force from the particle 
 *	    to the gas.
 *
 * Serves for the corrector step. It deals with one particle at a time.
 * Input: pG: grid with particles; gri,grf: initial and final particles;
 *        dv: velocity difference between gri and grf.
 * Output: pG: the array of drag forces exerted by the particle is updated
*/
void feedback_corrector(GridS *pG, GrainS *gri, GrainS *grf, Real3Vect cell1,
                                   Real dv1, Real dv2, Real dv3, Real ts)
{
  int is, js, ks;
  Real x1, x2, x3, v1, v2, v3;
  Real mgr;
  Real weight[3][3][3];
  Real Elosspar;                        /* particle energy dissipation */
  Real3Vect fb;

  mgr = grproperty[gri->property].m;
  x1 = 0.5*(gri->x1+grf->x1);
  x2 = 0.5*(gri->x2+grf->x2);
  x3 = 0.5*(gri->x3+grf->x3);
  v1 = 0.5*(gri->v1+grf->v1);
  v2 = 0.5*(gri->v2+grf->v2);
  v3 = 0.5*(gri->v3+grf->v3);

  /* Force other than drag force */
  fb = Get_Force(pG, x1, x2, x3, v1, v2, v3);

  fb.x1 = dv1 - pG->dt*fb.x1;
  fb.x2 = dv2 - pG->dt*fb.x2;
  fb.x3 = dv3 - pG->dt*fb.x3;

  /* energy dissipation */
  Elosspar = mgr*(SQR(fb.x1)+SQR(fb.x2)+SQR(fb.x3))*(ts/pG->dt);

  /* Drag force density */
  fb.x1 = mgr*fb.x1;
  fb.x2 = mgr*fb.x2;
  fb.x3 = mgr*fb.x3;

  /* distribute the drag force (density) to the grid */
  getweight(pG, x1, x2, x3, cell1, weight, &is, &js, &ks);
  distrFB_corr(pG, weight, is, js, ks, fb, Elosspar);

  return;

}

#endif /* FEEDBACK */


/*=========================== PRIVATE FUNCTIONS ==============================*/
/*----------------------------------------------------------------------------*/
/*! \fn void Delete_Ghost(GridS *pG)
 *  \brief Delete ghost particles */
void Delete_Ghost(GridS *pG)
{
  long p;
  GrainS *gr;

  p = 0;
  while (p<pG->nparticle)
  {/* loop over all particles */
    gr = &(pG->particle[p]);

    if (gr->pos == 0)
    {/* gr is a ghost particle */
      pG->nparticle -= 1;
      grproperty[gr->property].num -= 1;
      pG->particle[p] = pG->particle[pG->nparticle];
    }
    else
      p++;
  }

  return;
}

/*--------------------------------------------------------------------------- */
/*! \fn void JudgeCrossing(GridS *pG, Real x1, Real x2, Real x3, GrainS *gr)
 *  \brief Judge if the particle is a crossing particle */
void JudgeCrossing(GridS *pG, Real x1, Real x2, Real x3, GrainS *gr)
{
#ifndef FARGO
    /* if it crosses the grid boundary, mark it as a crossing out particle */
    if ((x1>=x1upar) || (x1<x1lpar) || (x2>=x2upar) || (x2<x2lpar) ||
                                       (x3>=x3upar) || (x3<x3lpar) )
      gr->pos = 10;
#else
    /* FARGO will naturally return the "crossing out" particles in the x2
        direction to the grid
     */
    if ((x1>=x1upar) || (x1<x1lpar) || (x3>=x3upar) || (x3<x3lpar))
      gr->pos = 10;
    else if ((ShBoxCoord == xz) && ((x2>=x2upar) || (x2<x2lpar)))
      gr->pos = 10;
#endif

    return;
}

/*--------------------------------------------------------------------------- */
/*! \fn Real3Vect Get_Drag(GridS *pG, int type, Real x1, Real x2, Real x3,
 *              Real v1, Real v2, Real v3, Real3Vect cell1, Real *tstop1)
 *  \brief Calculate the drag force to the particles 
 *
 * Input:
 *   pG: grid;	type: particle type;	cell1: 1/dx1,1/dx2,1/dx3;
 *   x1,x2,x3,v1,v2,v3: particle position and velocity;
 * Output:
 *   tstop1: 1/stopping time;
 * Return:
 *   drag force;
 */
Real3Vect Get_Drag(GridS *pG, int type, Real x1, Real x2, Real x3,
                Real v1, Real v2, Real v3, Real3Vect cell1, Real *tstop1)
{
  int is, js, ks;
  Real rho, u1, u2, u3, cs;
  Real vd1, vd2, vd3, vd, tstop, ts1;
#ifdef FEEDBACK
  Real stiffness;
#endif
  Real weight[3][3][3];		/* weight function */
  Real3Vect fd;

  /* interpolation to get fluid density, velocity and the sound speed */
  getweight(pG, x1, x2, x3, cell1, weight, &is, &js, &ks);

#ifndef FEEDBACK
  if (getvalues(pG, weight, is, js, ks, &rho, &u1, &u2, &u3, &cs) == 0)
#else
  if (getvalues(pG, weight, is, js, ks, &rho,&u1,&u2,&u3,&cs, &stiffness) == 0)
#endif
  { /* particle in the grid */

    /* apply possible gas velocity shift (e.g., for fake gas velocity field) */
    gasvshift(x1, x2, x3, &u1, &u2, &u3);

    /* velocity difference */
    vd1 = v1-u1;
    vd2 = v2-u2;
    vd3 = v3-u3;
    vd = sqrt(SQR(vd1) + SQR(vd2) + SQR(vd3)); /* dimension independent */

    /* particle stopping time */
    tstop = get_ts(pG, type, rho, cs, vd);
#ifdef FEEDBACK
    tstop *= MAX(1.0, stiffness);
#endif
    ts1 = 1.0/tstop;
  }
  else
  { /* particle out of the grid, free motion, with warning sign */
    vd1 = 0.0;	vd2 = 0.0;	vd3 = 0.0;	ts1 = 0.0;
    ath_perr(0, "Particle move out of grid %d with position (%f,%f,%f)!\n",
                                      myID_Comm_world,x1,x2,x3); /* warning! */
  }

  *tstop1 = ts1;

  /* Drag force */
  fd.x1 = -ts1*vd1;
  fd.x2 = -ts1*vd2;
  fd.x3 = -ts1*vd3;

  return fd;
}

/*--------------------------------------------------------------------------- */
/*! \fn Real3Vect Get_Force(GridS *pG, Real x1, Real x2, Real x3,
 *                                     Real v1, Real v2, Real v3)
 *  \brief Calculate the forces to the particle other than the gas drag
 *
 * Input:
 *   pG: grid;
 *   x1,x2,x3,v1,v2,v3: particle position and velocity;
 * Return:
 *   forces;
 */
Real3Vect Get_Force(GridS *pG, Real x1, Real x2, Real x3,
                               Real v1, Real v2, Real v3)
{
  Real3Vect ft;

  ft.x1 = ft.x2 = ft.x3 = 0.0;

/* User defined forces
 * Should be independent of velocity, or the integrators must be modified
 * Can also change the velocity to account for velocity difference.
 */
  Userforce_particle(&ft, x1, x2, x3, v1, v2, v3);

#ifdef SHEARING_BOX
  Real omg2 = SQR(Omega_0);

  if (ShBoxCoord == xy)
  {/* (x1,x2,x3)=(X,Y,Z) */
  #ifdef FARGO
    ft.x1 += 2.0*v2*Omega_0;
    ft.x2 += (qshear-2.0)*v1*Omega_0;
  #else
    ft.x1 += 2.0*(qshear*omg2*x1 + v2*Omega_0);
    ft.x2 += -2.0*v1*Omega_0;
  #endif /* FARGO */
  }
  else
  { /* (x1,x2,x3)=(X,Z,Y) */
  #ifdef FARGO
    ft.x1 += 2.0*v3*Omega_0;
    ft.x3 += (qshear-2.0)*v1*Omega_0;
  #else
    ft.x1 += 2.0*(qshear*omg2*x1 + v3*Omega_0);
    ft.x3 += -2.0*v1*Omega_0;
  #endif /* FARGO */
  }
#endif /* SHEARING_BOX */

  return ft;
}

#endif /*PARTICLES*/
