/*
 * Simulación numérica de la ecuación de onda 3D usando diferencias finitas
 *
 * Ecuación continua:
 *
 *     d²u/dt² = c² ( d²u/dx² + d²u/dy² + d²u/dz² )
 *
 * donde:
 *   - u = amplitud del campo/onda
 *   - c = velocidad de propagación
 *
 * Idea del programa:
 *   1. Se construye una grilla 3D uniforme.
 *   2. Se define una condición inicial tipo pulso gaussiano.
 *   3. Se hace evolucionar la onda en el tiempo con un esquema explícito.
 *   4. Se guarda un corte central del volumen en un archivo.
 *   5. Se mide el tiempo total y el tiempo de cálculo puro.
 *
 * Esto es útil porque más adelante, al paralelizar, se va a comparar:
 *   - versión secuencial
 *   - versión paralela
 * y calcular el ahorro de tiempo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * 1. PARÁMETROS DE LA MALLA Y DE LA SIMULACIÓN
 * ============================================================
 *
 * NX, NY, NZ  : número de puntos en cada dirección espacial
 * NSTEPS      : cantidad de pasos temporales
 * DX, DY, DZ  : separación espacial entre puntos
 * DT          : paso de tiempo
 * C           : velocidad de propagación
 */

#define NX 64
#define NY 64
#define NZ 64

#define NSTEPS 200

#define DX 0.01
#define DY 0.01
#define DZ 0.01

#define DT 0.005
#define C 1.0

/*
 * Macro para convertir índices 3D (i,j,k) en un índice 1D.
 *
 * Como la memoria se reserva como un solo arreglo lineal, necesitamos
 * una forma de mapear:
 *
 *   (i, j, k)  --->  posición dentro del arreglo lineal
 *
 * Esto evita tener que usar triple puntero y suele ser más eficiente.
 */
#define IDX(i,j,k) ((i)*(NY)*(NZ) + (j)*(NZ) + (k))

/* ============================================================
 * 2. PROTOTIPOS DE FUNCIONES
 * ============================================================
 */

double *alloc_grid(void);
void free_grid(double *g);

void init_gaussian(double *u, double cx, double cy, double cz, double sigma);
void apply_boundary_conditions(double *u);
void step(const double *u_prev, const double *u_curr, double *u_next);

double elapsed_seconds(struct timespec start, struct timespec end);

/* ============================================================
 * 3. FUNCIÓN PRINCIPAL
 * ============================================================
 */
int main()
{
    /*
     * Verificación de estabilidad tipo CFL.
     *
     * Para este esquema explícito en 3D, una condición típica de estabilidad es:
     *
     *   C * DT / DX <= 1 / sqrt(3)
     *
     * Aquí estamos suponiendo DX = DY = DZ.
     *
     * Si no se cumple, la simulación puede explotar numéricamente.
     */
    double r = C * DT / DX;

    if (r > 1.0 / sqrt(3.0))
    {
        printf("CFL violada\n");
        return 1;
    }

    /*
     * Reservamos memoria para tres estados temporales:
     *
     * u_prev : estado en el tiempo anterior
     * u_curr : estado actual
     * u_next : estado siguiente que vamos a calcular
     *
     * Esto se usa porque la ecuación tiene segunda derivada temporal,
     * así que para avanzar en el tiempo necesitamos dos estados previos.
     */
    double *u_prev = alloc_grid();
    double *u_curr = alloc_grid();
    double *u_next = alloc_grid();

    /*
     * Centro geométrico del dominio.
     *
     * Lo usamos para ubicar el pulso gaussiano inicial en el medio del cubo.
     */
    double cx = (NX - 1) * DX * 0.5;
    double cy = (NY - 1) * DY * 0.5;
    double cz = (NZ - 1) * DZ * 0.5;

    /*
     * Condición inicial:
     * ponemos un pulso gaussiano 3D centrado en el dominio.
     */
    init_gaussian(u_curr, cx, cy, cz, 0.05);

    /*
     * Velocidad inicial = 0
     *
     * Una forma estándar de imponer velocidad inicial nula es tomar:
     *
     *   u_prev = u_curr
     *
     * al comienzo de la simulación.
     *
     * Eso equivale a que inicialmente no haya cambio temporal.
     */
    memcpy(u_prev, u_curr, (size_t)NX * NY * NZ * sizeof(double));

    /*
     * Abrimos el archivo de salida donde guardaremos cortes 2D
     * del volumen 3D.
     *
     * Guardaremos los datos en data/wave.dat
     */
    FILE *f = fopen("data/wave.dat", "w");

    if (!f)
    {
        perror("fopen");
        return 1;
    }

    /*
     * Variables para medir tiempo:
     *
     * total_start / total_end : miden todo el bucle temporal
     * step_start / step_end   : miden solo el costo de step(...)
     *
     */
    struct timespec total_start, total_end;
    struct timespec step_start, step_end;

    double compute_time = 0.0;

    /*
     * Empezamos a medir el tiempo total de la simulación.
     */
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    /*
     * Bucle temporal principal.
     *
     * En cada iteración:
     *   1. calculamos u_next
     *   2. guardamos datos cada cierto número de pasos
     *   3. rotamos punteros para avanzar al siguiente tiempo
     */
    for (int t = 1; t <= NSTEPS; t++)
    {
        /*
         * Medimos solo el tiempo de cálculo del paso numérico.
         */
        clock_gettime(CLOCK_MONOTONIC, &step_start);
        step(u_prev, u_curr, u_next);
        clock_gettime(CLOCK_MONOTONIC, &step_end);

        compute_time += elapsed_seconds(step_start, step_end);

        /*
         * Guardamos un corte central cada 10 pasos temporales.
         *
         * Elegimos el plano z = NZ/2.
         *
         * Es decir: del volumen 3D completo, guardamos solo una "lámina" 2D.
         * Eso hace más fácil visualizar la evolución con Python.
         *
         * Formato por línea:
         *   i j valor
         *
         * Entre snapshots dejamos una línea en blanco para que Python
         * pueda separar los frames de la animación.
         */
        if (t % 10 == 0)
        {
            int k = NZ / 2;

            for (int i = 0; i < NX; i++)
            {
                for (int j = 0; j < NY; j++)
                {
                    fprintf(f, "%d %d %.10f\n",
                            i, j,
                            u_curr[IDX(i, j, k)]);
                }
            }

            fprintf(f, "\n");
        }

        /*
         * Rotación de punteros:
         *
         * Antes:
         *   u_prev = tiempo n-1
         *   u_curr = tiempo n
         *   u_next = tiempo n+1 calculado recién
         *
         * Después de rotar:
         *   u_prev pasa a ser el viejo u_curr
         *   u_curr pasa a ser el viejo u_next
         *   u_next queda disponible para reutilizarse
         *
         * Esto evita copiar arreglos gigantes en cada paso.
         */
        double *tmp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = tmp;
    }

    /*
     * Terminamos de medir el tiempo total.
     */
    clock_gettime(CLOCK_MONOTONIC, &total_end);

    fclose(f);

    /*
     * Convertimos el tiempo total a segundos.
     */
    double total_time = elapsed_seconds(total_start, total_end);

    /*
     * Mostramos tiempos relevantes:
     *
     * - Tiempo total: incluye cálculo + escritura de archivo
     * - Tiempo de cálculo: solo la rutina numérica
     * - Promedio por step: útil para comparar escalamiento
     */
    printf("Tiempo total             : %.6f s\n", total_time);
    printf("Tiempo de calculo (step) : %.6f s\n", compute_time);
    printf("Tiempo promedio por step : %.6f s\n", compute_time / NSTEPS);

    /*
     * Liberamos memoria reservada dinámicamente.
     */
    free_grid(u_prev);
    free_grid(u_curr);
    free_grid(u_next);

    return 0;
}

/* ============================================================
 * 4. PASO TEMPORAL
 * ============================================================
 *
 * Esta función implementa la discretización explícita:
 *
 *   u_next = 2 u_curr - u_prev + c² DT² ∇²u_curr
 *
 * donde el laplaciano ∇²u se aproxima con diferencias centrales
 * en x, y y z.
 */
void step(const double *u_prev, const double *u_curr, double *u_next)
{
    /*
     * Factor común que aparece en el esquema temporal.
     */
    double c2dt2 = C * C * DT * DT;

    /*
     * Recorremos solo los puntos interiores.
     *
     * Los bordes no se calculan aquí porque se fijan aparte
     * con condiciones de contorno.
     */
    for (int i = 1; i < NX - 1; i++)
    {
        for (int j = 1; j < NY - 1; j++)
        {
            for (int k = 1; k < NZ - 1; k++)
            {
                /*
                 * Segunda derivada respecto de x:
                 *
                 * d²u/dx² ≈ (u(i+1)-2u(i)+u(i-1))/DX²
                 */
                double d2x =
                    (u_curr[IDX(i + 1, j, k)]
                     - 2.0 * u_curr[IDX(i, j, k)]
                     + u_curr[IDX(i - 1, j, k)]) / (DX * DX);

                /*
                 * Segunda derivada respecto de y.
                 */
                double d2y =
                    (u_curr[IDX(i, j + 1, k)]
                     - 2.0 * u_curr[IDX(i, j, k)]
                     + u_curr[IDX(i, j - 1, k)]) / (DY * DY);

                /*
                 * Segunda derivada respecto de z.
                 */
                double d2z =
                    (u_curr[IDX(i, j, k + 1)]
                     - 2.0 * u_curr[IDX(i, j, k)]
                     + u_curr[IDX(i, j, k - 1)]) / (DZ * DZ);

                /*
                 * Laplaciano 3D:
                 *
                 * ∇²u = d²u/dx² + d²u/dy² + d²u/dz²
                 */
                double lap = d2x + d2y + d2z;

                /*
                 * Actualización temporal del esquema de segundo orden.
                 */
                u_next[IDX(i, j, k)] =
                    2.0 * u_curr[IDX(i, j, k)]
                    - u_prev[IDX(i, j, k)]
                    + c2dt2 * lap;
            }
        }
    }

    /*
     * Una vez calculado el nuevo estado, imponemos condiciones de borde.
     */
    apply_boundary_conditions(u_next);
}

/* ============================================================
 * 5. CONDICIONES DE CONTORNO
 * ============================================================
 *
 * Aquí imponemos condiciones de Dirichlet homogéneas:
 *
 *   u = 0
 *
 * en todas las caras del cubo.
 *
 * Físicamente esto puede interpretarse como bordes fijos.
 */
void apply_boundary_conditions(double *u)
{
    /*
     * Caras x = 0 y x = Lx
     */
    for (int j = 0; j < NY; j++)
    {
        for (int k = 0; k < NZ; k++)
        {
            u[IDX(0, j, k)] = 0.0;
            u[IDX(NX - 1, j, k)] = 0.0;
        }
    }

    /*
     * Caras y = 0 y y = Ly
     */
    for (int i = 0; i < NX; i++)
    {
        for (int k = 0; k < NZ; k++)
        {
            u[IDX(i, 0, k)] = 0.0;
            u[IDX(i, NY - 1, k)] = 0.0;
        }
    }

    /*
     * Caras z = 0 y z = Lz
     */
    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NY; j++)
        {
            u[IDX(i, j, 0)] = 0.0;
            u[IDX(i, j, NZ - 1)] = 0.0;
        }
    }
}

/* ============================================================
 * 6. CONDICIÓN INICIAL: PULSO GAUSSIANO
 * ============================================================
 *
 * Definimos:
 *
 *   u(x,y,z) = exp( -r² / (2 sigma²) )
 *
 * con:
 *
 *   r² = (x-cx)² + (y-cy)² + (z-cz)²
 *
 * Esto genera una perturbación localizada en el centro del dominio.
 */
void init_gaussian(double *u, double cx, double cy, double cz, double sigma)
{
    double s2 = sigma * sigma;

    for (int i = 0; i < NX; i++)
    {
        double x = i * DX;

        for (int j = 0; j < NY; j++)
        {
            double y = j * DY;

            for (int k = 0; k < NZ; k++)
            {
                double z = k * DZ;

                double d2 =
                    (x - cx) * (x - cx) +
                    (y - cy) * (y - cy) +
                    (z - cz) * (z - cz);

                u[IDX(i, j, k)] = exp(-d2 / (2.0 * s2));
            }
        }
    }
}

/* ============================================================
 * 7. RESERVA DE MEMORIA
 * ============================================================
 *
 * Reserva memoria para una grilla linealizada de tamaño NX*NY*NZ.
 */
double *alloc_grid(void)
{
    double *g = malloc((size_t)NX * NY * NZ * sizeof(double));

    if (!g)
    {
        perror("malloc");
        exit(1);
    }

    return g;
}

/* ============================================================
 * 8. LIBERACIÓN DE MEMORIA
 * ============================================================
 */
void free_grid(double *g)
{
    free(g);
}

/* ============================================================
 * 9. CÁLCULO DE TIEMPO TRANSCURRIDO
 * ============================================================
 *
 * Convierte dos timestamps en un intervalo en segundos.
 */
double elapsed_seconds(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec)
         + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}
