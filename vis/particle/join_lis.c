/*==============================================================================
 * FILE: join_lis.c
 *
 * PURPOSE: Joins together multiple particle list files generated by an MPI job
 *   into one file for visualization and analysis.
 *
 * COMPILE USING: gcc -Wall -W -o join_lis join_lis.c -lm
 *
 * USAGE: ./join_lis -p <nproc> -o <basename-out> -i <basename-in> -s <post-name>
 *                   -d <outdir> -f <# range(f1:f2)>
 *
 * WRITTEN BY: Xuening Bai, September 2009
 *============================================================================*/

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <assert.h>

static void join_error(const char *fmt, ...);
static void usage(const char *arg);

/* ========================================================================== */

int main(int argc, char* argv[])
{
  /* argument variables */
  int nproc=0,f1=0,f2=0,fi=1;
  char *defoutdir = "comb_lis";
  char *outbase = NULL, *inbase = NULL, *postname = NULL;
  char *outdir = defoutdir;
  /* fild variables */
  FILE *fidin,*fidout;
  char in_name[100], out_name[100];
  struct stat st;
  /* data variables */
  int i,p,err,cpuid,ntype,property;
  long j,n,ntot,pid;
  float time[2],buffer[20],*typeinfo;
  short shock_of_origin, injected;

  /* Read Arguments */
  for (i=1; i<argc; i++) {
/* If argv[i] is a 2 character string of the form "-?" then: */
    if(*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0'){
      switch(*(argv[i]+1)) {
      case 'p':                                /* -i <nproc>   */
        nproc = atoi(argv[++i]);
        break;
      case 'i':                                /* -d <basename-in>   */
        inbase = argv[++i];
        break;
      case 'f':                                /* -f <# range(f1:f2:fi)>*/
        sscanf(argv[++i],"%d:%d:%d",&f1,&f2,&fi);
        if (f2 == 0) f2 = f1;
        break;
      case 's':                                /* -s <post-name> */
        postname = argv[++i];
        break;
      case 'd':                                /* -d <outdir> */
        outdir = argv[++i];
        break;
      case 'o':                                /* -o <basename-out>   */
        outbase = argv[++i];
        break;
      case 'h':                                /* -h */
        usage(argv[0]);
        break;
      default:
        usage(argv[0]);
        break;
      }
    }
  }

  /* Checkpoints */
  if (nproc <= 0)
    join_error("Please specify number of processors by using -p option!\n");

  if (inbase == NULL)
    join_error("Please specify input file basename using -i option!\n");

  if (outbase == NULL)
    outbase = inbase;

  if (postname == NULL)
    join_error("Please specify posterior file name using -s option!\n");

  if ((f1>f2) || (f2<0) || (fi<=0))
    join_error("Wrong number sequence in the -f option!\n");

  /* Check output directory */
  if(stat(outdir,&st) != 0) /* output directory does not exist */
  {
    err = mkdir(outdir, S_IRWXU | S_IRWXG | S_IRWXO);
    if (err != 0) /* mkdir fails */
      join_error("Fail to make output directory!\n");
  }

  fprintf(stderr,"Output combined files to %s/\n",outdir);

  /* ====================================================================== */

  for (i=f1; i<=f2; i+=fi)
  {
    fprintf(stderr,"Processing file number %d...\n",i);

    /* Step 1: Count the total # of particles */
    ntot = 0;
    for (p=0; p<nproc; p++)
    {
      if (p == 0)
        sprintf(in_name,"id%d/%s.%04d.%s.lis",p,inbase,i,postname);
      else
        sprintf(in_name,"id%d/%s-id%d.%04d.%s.lis",p,inbase,p,i,postname);

      fidin = fopen(in_name,"rb");
      if (fidin == NULL)
        join_error("Fail to open input file %s!\n",in_name);

      fseek(fidin, 12*sizeof(float), SEEK_SET);
      fread(&ntype,sizeof(int),1,fidin);
      fseek(fidin, ntype*sizeof(float), SEEK_CUR);
      fread(buffer,sizeof(float),2,fidin);
      fread(&n,sizeof(long),1,fidin);

      ntot += n;

      fclose(fidin);
    }

    fprintf(stderr,"ntot=%ld\n",ntot);

    /* Step 2: Read input and write output */
    sprintf(out_name,"%s/%s.%04d.%s.lis",outdir,outbase,i,postname);

    fidout = fopen(out_name,"wb");
    if (fidout == NULL)
        join_error("Fail to open output file %s!\n",out_name);

    typeinfo = (float*)calloc(ntype,sizeof(float));

    for (p=0; p<nproc; p++)
    {
      if (p == 0)
        sprintf(in_name,"id%d/%s.%04d.%s.lis",p,inbase,i,postname);
      else
        sprintf(in_name,"id%d/%s-id%d.%04d.%s.lis",p,inbase,p,i,postname);

      fidin = fopen(in_name,"rb");

      /* read header */
      fread(buffer,sizeof(float),12,fidin);
      fread(&ntype,sizeof(int),1,fidin);
      fread(typeinfo,sizeof(float),ntype,fidin);
      fread(time,sizeof(float),2,fidin);
      fread(&n,sizeof(long),1,fidin);

      /* write header */
      if (p == 0)
      {
        for (j=0; j<6; j++)
          buffer[j] = buffer[j+6];

        fwrite(buffer,sizeof(float),12,fidout);
        fwrite(&ntype,sizeof(int),1,fidout);
        fwrite(typeinfo,sizeof(float),ntype,fidout);
        fwrite(time,sizeof(float),2,fidout);
        fwrite(&ntot,sizeof(long),1,fidout);
      }

      /* read and write data */
      for (j=0; j<n; j++)
      {
        fread(buffer,sizeof(float),7,fidin);
        fread(&property,sizeof(int),1,fidin);
        fread(&pid,sizeof(long),1,fidin);
        fread(&cpuid,sizeof(int),1,fidin);
        fread(&shock_of_origin,sizeof(short),1,fidin);
        fread(&injected,sizeof(short),1,fidin);

        fwrite(buffer,sizeof(float),7,fidout);
        fwrite(&property,sizeof(int),1,fidout);
        fwrite(&pid,sizeof(long),1,fidout);
        fwrite(&cpuid,sizeof(int),1,fidout);
        fwrite(&shock_of_origin,sizeof(short),1,fidout);
        fwrite(&injected,sizeof(short),1,fidout);

        // [PP] additional quantities

      }

      fclose(fidin);
    }

    fclose(fidout);

    if (typeinfo != NULL)
      free(typeinfo);
  }

  return 0;
}


/* ========================================================================== */


/* Write an error message and terminate the simulation with an error status. */
static void join_error(const char *fmt, ...){
  va_list ap;

  va_start(ap, fmt);         /* ap starts after the fmt parameter */
  vfprintf(stderr, fmt, ap); /* print the error message to stderr */
  va_end(ap);                /* end stdargs (clean up the va_list ap) */

  fflush(stderr);            /* flush it NOW */
  exit(1);                   /* clean up and exit */
}

static void usage(const char *arg)
{
  fprintf(stderr,"\nUsage: %s [options] [block] ...\n", arg);
  fprintf(stderr,"\nOptions:\n");
  fprintf(stderr,"  -p nproc        number of processors\n");
  fprintf(stderr,"  -i <name>       basename of input file\n");
  fprintf(stderr,"  -s <name>       posterior name of input file\n");
  fprintf(stderr,"  -f f1:f2:fi     file number range and interval\n");
  fprintf(stderr,"                  Default: <0:0:1>\n");
  fprintf(stderr,"  -o <name>       basename of output file\n");
  fprintf(stderr,"                  Default: <input file basename>\n");
  fprintf(stderr,"  -d <directory>  name of the output directory\n");
  fprintf(stderr,"                  Default: <comb_lis>\n");
  fprintf(stderr,"  -h              this help\n");

  fprintf(stderr,"\nExample:\n");
  fprintf(stderr,"%s -p 64 -i streaming2d -s ds -f 0:500\n\n", arg);

  exit(0);
}
