/*
 * Simulación de la ecuación de onda 3D mediante diferencias finitas
 *
 * d²u/dt² = c² ( d²u/dx² + d²u/dy² + d²u/dz² )
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* --- Parámetros de la grilla --- */

#define NX 64
#define NY 64
#define NZ 64

#define NSTEPS 200

#define DX 0.01
#define DY 0.01
#define DZ 0.01

#define DT 0.005
#define C 1.0

#define IDX(i,j,k) ((i)*(NY)*(NZ) + (j)*(NZ) + (k))

/* prototipos */

double *alloc_grid(void);
void free_grid(double *g);

void init_gaussian(double *u,double cx,double cy,double cz,double sigma);

void apply_boundary_conditions(double *u);

void step(const double *u_prev,const double *u_curr,double *u_next);

double elapsed_seconds(struct timespec start, struct timespec end);

int main()
{
    double r = C*DT/DX;

    if(r > 1.0/sqrt(3.0))
    {
        printf("CFL violada\n");
        return 1;
    }

    double *u_prev = alloc_grid();
    double *u_curr = alloc_grid();
    double *u_next = alloc_grid();

    double cx = (NX-1)*DX*0.5;
    double cy = (NY-1)*DY*0.5;
    double cz = (NZ-1)*DZ*0.5;

    init_gaussian(u_curr,cx,cy,cz,0.05);

    memcpy(u_prev,u_curr,(size_t)NX*NY*NZ*sizeof(double));

    FILE *f = fopen("wave.dat","w");

    if(!f)
    {
        perror("fopen");
        return 1;
    }

    struct timespec total_start, total_end;
    struct timespec step_start, step_end;

    double compute_time = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for(int t=1;t<=NSTEPS;t++)
    {
        clock_gettime(CLOCK_MONOTONIC, &step_start);
        step(u_prev,u_curr,u_next);
        clock_gettime(CLOCK_MONOTONIC, &step_end);

        compute_time += elapsed_seconds(step_start, step_end);

        if(t%10==0)
        {
            int k = NZ/2;

            for(int i=0;i<NX;i++)
            {
                for(int j=0;j<NY;j++)
                {
                    fprintf(f,"%d %d %f\n",
                           i,j,
                           u_curr[IDX(i,j,k)]);
                }
            }

            fprintf(f,"\n");
        }

        double *tmp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);

    fclose(f);

    double total_time = elapsed_seconds(total_start, total_end);

    printf("Tiempo total            : %.6f s\n", total_time);
    printf("Tiempo de calculo (step): %.6f s\n", compute_time);
    printf("Tiempo promedio por step: %.6f s\n", compute_time / NSTEPS);

    free_grid(u_prev);
    free_grid(u_curr);
    free_grid(u_next);

    return 0;
}

void step(const double *u_prev,const double *u_curr,double *u_next)
{
    double c2dt2 = C*C*DT*DT;

    for(int i=1;i<NX-1;i++)
    {
        for(int j=1;j<NY-1;j++)
        {
            for(int k=1;k<NZ-1;k++)
            {
                double d2x =
                (u_curr[IDX(i+1,j,k)]
                -2*u_curr[IDX(i,j,k)]
                +u_curr[IDX(i-1,j,k)])/(DX*DX);

                double d2y =
                (u_curr[IDX(i,j+1,k)]
                -2*u_curr[IDX(i,j,k)]
                +u_curr[IDX(i,j-1,k)])/(DY*DY);

                double d2z =
                (u_curr[IDX(i,j,k+1)]
                -2*u_curr[IDX(i,j,k)]
                +u_curr[IDX(i,j,k-1)])/(DZ*DZ);

                double lap = d2x + d2y + d2z;

                u_next[IDX(i,j,k)] =
                2*u_curr[IDX(i,j,k)]
                -u_prev[IDX(i,j,k)]
                +c2dt2*lap;
            }
        }
    }

    apply_boundary_conditions(u_next);
}

void apply_boundary_conditions(double *u)
{
    for(int j=0;j<NY;j++)
    for(int k=0;k<NZ;k++)
    {
        u[IDX(0,j,k)] = 0.0;
        u[IDX(NX-1,j,k)] = 0.0;
    }

    for(int i=0;i<NX;i++)
    for(int k=0;k<NZ;k++)
    {
        u[IDX(i,0,k)] = 0.0;
        u[IDX(i,NY-1,k)] = 0.0;
    }

    for(int i=0;i<NX;i++)
    for(int j=0;j<NY;j++)
    {
        u[IDX(i,j,0)] = 0.0;
        u[IDX(i,j,NZ-1)] = 0.0;
    }
}

void init_gaussian(double *u,double cx,double cy,double cz,double sigma)
{
    double s2 = sigma*sigma;

    for(int i=0;i<NX;i++)
    {
        double x = i*DX;

        for(int j=0;j<NY;j++)
        {
            double y = j*DY;

            for(int k=0;k<NZ;k++)
            {
                double z = k*DZ;

                double d2 =
                (x-cx)*(x-cx)
                +(y-cy)*(y-cy)
                +(z-cz)*(z-cz);

                u[IDX(i,j,k)] = exp(-d2/(2*s2));
            }
        }
    }
}

double *alloc_grid()
{
    double *g = malloc((size_t)NX*NY*NZ*sizeof(double));

    if(!g)
    {
        perror("malloc");
        exit(1);
    }

    return g;
}

void free_grid(double *g)
{
    free(g);
}

double elapsed_seconds(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec)
         + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}
