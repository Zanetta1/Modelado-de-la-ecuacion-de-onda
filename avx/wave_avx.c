/*
 * Simulación numérica de la ecuación de onda 3D usando diferencias finitas
 * y vectorización AVX en el paso temporal.
 *
 * Ecuación continua:
 *
 *     d²u/dt² = c² ( d²u/dx² + d²u/dy² + d²u/dz² )
 *
 * ============================================================
 * IDEA DE ESTA VERSIÓN
 * ============================================================
 *
 * Esta versión conserva la misma lógica física y numérica
 * de la versión escalar original, pero reemplaza parte del cálculo
 * del paso temporal por instrucciones AVX.
 *
 * IMPORTANTE:
 * - AVX aplica SIMD: una instrucción opera sobre varios datos.
 *
 * En double, AVX de 256 bits permite procesar 4 valores a la vez.
 *
 * La vectorización se aplica sobre el índice k porque, con la forma
 * en que está almacenada la grilla, los puntos consecutivos en k
 * están contiguos en memoria.
 *
 * ============================================================
 * CAMBIOS PRINCIPALES RESPECTO A LA VERSIÓN ESCALAR
 * ============================================================
 *
 * CAMBIO 1:
 *   Se agrega:
 *       #include <immintrin.h>
 *   para poder usar instrucciones intrínsecas AVX.
 *
 * CAMBIO 2:
 *   Se reemplaza la función step(...) por una versión vectorizada.
 *
 * CAMBIO 3:
 *   Se guardan los resultados en:
 *       data/wave_avx.dat
 *   para no sobreescribir la salida de la versión escalar.
 *
 * CAMBIO 4:
 *   Se imprime que esta es la versión AVX para no confundirse
 *   al comparar tiempos.
 *
 * ============================================================
 * QUÉ NO CAMBIA
 * ============================================================
 *
 * - La ecuación física
 * - La discretización
 * - La condición inicial
 * - Las condiciones de borde
 * - La condición CFL
 * - El formato general del archivo de salida
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>   /* CAMBIO AVX 1: habilita intrínsecos AVX */

/* ============================================================
 * 1. PARÁMETROS DE LA MALLA Y DE LA SIMULACIÓN
 * ============================================================
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
 * Mapeo de índices 3D a índice 1D.
 *
 */
#define IDX(i,j,k) ((i)*(NY)*(NZ) + (j)*(NZ) + (k))

/* ============================================================
 * 2. PROTOTIPOS
 * ============================================================
 */

double *alloc_grid(void);
void free_grid(double *g);

void init_gaussian(double *u, double cx, double cy, double cz, double sigma);
void apply_boundary_conditions(double *u);

/*
 * Aquí es donde se usa AVX.
 */
void step_avx(const double *u_prev, const double *u_curr, double *u_next);

double elapsed_seconds(struct timespec start, struct timespec end);

/* ============================================================
 * 3. FUNCIÓN PRINCIPAL
 * ============================================================
 */
int main()
{
    /*
     * Verificación CFL para estabilidad numérica.
     */
    double r = C * DT / DX;

    if (r > 1.0 / sqrt(3.0))
    {
        printf("CFL violada\n");
        return 1;
    }

    /*
     * Reserva de memoria para los tres niveles temporales.
     */
    double *u_prev = alloc_grid();
    double *u_curr = alloc_grid();
    double *u_next = alloc_grid();

    /*
     * Centro geométrico del dominio.
     */
    double cx = (NX - 1) * DX * 0.5;
    double cy = (NY - 1) * DY * 0.5;
    double cz = (NZ - 1) * DZ * 0.5;

    /*
     * Condición inicial: pulso gaussiano.
     */
    init_gaussian(u_curr, cx, cy, cz, 0.05);

    /*
     * Velocidad inicial nula:
     * usamos u_prev = u_curr al inicio.
     */
    memcpy(u_prev, u_curr, (size_t)NX * NY * NZ * sizeof(double));

    /*
     * CAMBIO AVX 3:
     */
    FILE *f = fopen("data/wave_avx.dat", "w");

    if (!f)
    {
        perror("fopen");
        return 1;
    }

    struct timespec total_start, total_end;
    struct timespec step_start, step_end;

    double compute_time = 0.0;

    /*
     * CAMBIO AVX 4:
     */
    printf("Ejecutando version AVX\n");

    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int t = 1; t <= NSTEPS; t++)
    {
        /*
         * Medimos solo el costo del paso numérico.
         */
        clock_gettime(CLOCK_MONOTONIC, &step_start);
        step_avx(u_prev, u_curr, u_next);
        clock_gettime(CLOCK_MONOTONIC, &step_end);

        compute_time += elapsed_seconds(step_start, step_end);

        /*
         * Guardamos un corte central cada 10 pasos.
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
         * Rotación de punteros para avanzar en el tiempo.
         */
        double *tmp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);

    fclose(f);

    double total_time = elapsed_seconds(total_start, total_end);

    printf("Tiempo total             : %.6f s\n", total_time);
    printf("Tiempo de calculo (step) : %.6f s\n", compute_time);
    printf("Tiempo promedio por step : %.6f s\n", compute_time / NSTEPS);

    free_grid(u_prev);
    free_grid(u_curr);
    free_grid(u_next);

    return 0;
}

/* ============================================================
 * 4. PASO TEMPORAL VECTORIAL CON AVX
 * ============================================================
 *
 * Esta función reemplaza a la función step escalar.
 *
 * En la versión original, el bucle interno recorría k de uno en uno.
 *
 * Ahora:
 * - el bucle interno avanza de 4 en 4
 * - se cargan 4 doubles simultáneamente
 * - se calcula el laplaciano para 4 puntos a la vez
 * - se guarda u_next para esos 4 puntos juntos
 *
 * ============================================================
 * POR QUÉ SE VECTORIZA EN k
 * ============================================================
 *
 * Porque los índices consecutivos en k son contiguos en memoria:
 *
 *   u(i,j,k), u(i,j,k+1), u(i,j,k+2), u(i,j,k+3)
 *
 * Eso permite usar cargas vectoriales eficientes.
 * Porque NZ-2 no necesariamente es múltiplo de 4.
 * Entonces:
 *
 * - una parte se hace con AVX
 * - los últimos puntos sobrantes se hacen de forma escalar
 *
 * Así no se sale del arreglo ni se pierden puntos.
 */
void step_avx(const double *u_prev, const double *u_curr, double *u_next)
{
    /*
     * Precomputamos factores escalares para evitar repetir divisiones.
     *
     * Esto también ayuda a que la fórmula quede más limpia.
     */
    const double inv_dx2 = 1.0 / (DX * DX);
    const double inv_dy2 = 1.0 / (DY * DY);
    const double inv_dz2 = 1.0 / (DZ * DZ);
    const double c2dt2 = C * C * DT * DT;

    /*
     * CAMBIO AVX 2A:
     * Convertimos constantes escalares en vectores AVX.
     *
     * __m256d = vector AVX de 4 doubles.
     *
     * _mm256_set1_pd(x) crea:
     *   [x, x, x, x]
     *
     * Esto es necesario porque las operaciones AVX trabajan
     * entre vectores, no entre un vector y un double común.
     */
    const __m256d v_two   = _mm256_set1_pd(2.0);
    const __m256d v_dx2i  = _mm256_set1_pd(inv_dx2);
    const __m256d v_dy2i  = _mm256_set1_pd(inv_dy2);
    const __m256d v_dz2i  = _mm256_set1_pd(inv_dz2);
    const __m256d v_c2dt2 = _mm256_set1_pd(c2dt2);

    /*
     * Los bucles externos en i y j se mantienen iguales.
     * El cambio importante ocurre dentro de k.
     */
    for (int i = 1; i < NX - 1; i++)
    {
        for (int j = 1; j < NY - 1; j++)
        {
            /*
             * CAMBIO AVX 2B:
             * Creamos punteros a "filas" en memoria.
             *
             * Esto se hace por dos motivos:
             *
             * 1. simplifica mucho las expresiones
             * 2. evita recalcular IDX(i,j,k) todo el tiempo
             *
             * row_curr[k] representa u_curr(i,j,k)
             * row_ip1[k]  representa u_curr(i+1,j,k)
             * row_im1[k]  representa u_curr(i-1,j,k)
             * etc.
             */
            const double *row_prev = &u_prev[IDX(i, j, 0)];
            const double *row_curr = &u_curr[IDX(i, j, 0)];
            const double *row_im1  = &u_curr[IDX(i - 1, j, 0)];
            const double *row_ip1  = &u_curr[IDX(i + 1, j, 0)];
            const double *row_jm1  = &u_curr[IDX(i, j - 1, 0)];
            const double *row_jp1  = &u_curr[IDX(i, j + 1, 0)];
            double *row_next       = &u_next[IDX(i, j, 0)];

            /*
             * Empezamos en k = 1 porque k = 0 es borde.
             */
            int k = 1;

            /*
             * CAMBIO AVX 2C:
             * Bucle vectorial.
             *
             * Procesamos 4 posiciones de k por iteración.
             *
             * La condición k <= NZ - 5 garantiza que:
             * - podemos leer row_curr[k+1]...row_curr[k+4]
             * - no salimos del arreglo al construir vecinos
             */
            for (; k <= NZ - 5; k += 4)
            {
                /*
                 * Cargamos 4 valores consecutivos del estado actual
                 * y del estado previo.
                 *
                 * Usamos loadu en vez de load alineado porque no
                 * estamos imponiendo alineación explícita de 32 bytes.
                 *
                 * Esto hace la versión más robusta y más simple
                 * para empezar.
                 */
                __m256d curr = _mm256_loadu_pd(&row_curr[k]);
                __m256d prev = _mm256_loadu_pd(&row_prev[k]);

                /*
                 * Vecinos en dirección x:
                 *   (i-1,j,k...k+3) y (i+1,j,k...k+3)
                 */
                __m256d xim1 = _mm256_loadu_pd(&row_im1[k]);
                __m256d xip1 = _mm256_loadu_pd(&row_ip1[k]);

                /*
                 * Vecinos en dirección y:
                 *   (i,j-1,k...k+3) y (i,j+1,k...k+3)
                 */
                __m256d yjm1 = _mm256_loadu_pd(&row_jm1[k]);
                __m256d yjp1 = _mm256_loadu_pd(&row_jp1[k]);

                /*
                 * Vecinos en dirección z:
                 *
                 * Ojo aquí:
                 * para cada uno de los 4 puntos necesitamos
                 * el vecino anterior y el siguiente en k.
                 *
                 * Por eso cargamos:
                 *   row_curr[k-1 ... k+2]
                 *   row_curr[k+1 ... k+4]
                 *
                 * lane por lane esto corresponde a:
                 *
                 * zkm1 = [u(k-1), u(k),   u(k+1), u(k+2)]
                 * zkp1 = [u(k+1), u(k+2), u(k+3), u(k+4)]
                 *
                 * que es justo lo que necesita la derivada segunda.
                 */
                __m256d zkm1 = _mm256_loadu_pd(&row_curr[k - 1]);
                __m256d zkp1 = _mm256_loadu_pd(&row_curr[k + 1]);

                /*
                 * Vector [2u, 2u, 2u, 2u] lane por lane.
                 */
                __m256d two_curr = _mm256_mul_pd(v_two, curr);

                /*
                 * CAMBIO AVX 2D:
                 * d2x vectorial.
                 *
                 * Fórmula escalar:
                 *   (u(i+1)-2u(i)+u(i-1)) / DX²
                 *
                 * Aquí se hace exactamente igual, pero sobre 4 valores.
                 */
                __m256d d2x = _mm256_mul_pd(
                    _mm256_add_pd(_mm256_sub_pd(xip1, two_curr), xim1),
                    v_dx2i
                );

                /*
                 * d2y vectorial.
                 */
                __m256d d2y = _mm256_mul_pd(
                    _mm256_add_pd(_mm256_sub_pd(yjp1, two_curr), yjm1),
                    v_dy2i
                );

                /*
                 * d2z vectorial.
                 */
                __m256d d2z = _mm256_mul_pd(
                    _mm256_add_pd(_mm256_sub_pd(zkp1, two_curr), zkm1),
                    v_dz2i
                );

                /*
                 * Laplaciano vectorial:
                 *   lap = d2x + d2y + d2z
                 */
                __m256d lap = _mm256_add_pd(_mm256_add_pd(d2x, d2y), d2z);

                /*
                 * CAMBIO AVX 2E:
                 * actualización temporal vectorial:
                 *
                 * next = 2*curr - prev + c²*dt²*lap
                 */
                __m256d next = _mm256_add_pd(
                    _mm256_sub_pd(two_curr, prev),
                    _mm256_mul_pd(v_c2dt2, lap)
                );

                /*
                 * Guardamos 4 resultados consecutivos.
                 */
                _mm256_storeu_pd(&row_next[k], next);
            }

            /*
             * CAMBIO AVX 2F:
             * Cola escalar.
             *
             * Los últimos puntos interiores que no alcanzan a formar
             * un bloque de 4 se calculan con el método original.
             *
             * Esto es normal en código SIMD.
             */
            for (; k < NZ - 1; k++)
            {
                double d2x =
                    (row_ip1[k]
                    - 2.0 * row_curr[k]
                    + row_im1[k]) * inv_dx2;

                double d2y =
                    (row_jp1[k]
                    - 2.0 * row_curr[k]
                    + row_jm1[k]) * inv_dy2;

                double d2z =
                    (row_curr[k + 1]
                    - 2.0 * row_curr[k]
                    + row_curr[k - 1]) * inv_dz2;

                double lap = d2x + d2y + d2z;

                row_next[k] =
                    2.0 * row_curr[k]
                    - row_prev[k]
                    + c2dt2 * lap;
            }
        }
    }

    /*
     * Las condiciones de borde se mantienen exactamente igual.
     * No se vectorizan porque el costo es pequeño y la lógica es simple.
     */
    apply_boundary_conditions(u_next);
}

/* ============================================================
 * 5. CONDICIONES DE BORDE
 * ============================================================
 *
 * Igual que en la versión original:
 * u = 0 en todas las caras del dominio.
 */
void apply_boundary_conditions(double *u)
{
    for (int j = 0; j < NY; j++)
    {
        for (int k = 0; k < NZ; k++)
        {
            u[IDX(0, j, k)] = 0.0;
            u[IDX(NX - 1, j, k)] = 0.0;
        }
    }

    for (int i = 0; i < NX; i++)
    {
        for (int k = 0; k < NZ; k++)
        {
            u[IDX(i, 0, k)] = 0.0;
            u[IDX(i, NY - 1, k)] = 0.0;
        }
    }

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
 * 6. CONDICIÓN INICIAL
 * ============================================================
 *
 * Pulso gaussiano centrado en el dominio.
 * Igual que en la versión original.
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
 * 7. RESERVA Y LIBERACIÓN DE MEMORIA
 * ============================================================
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

void free_grid(double *g)
{
    free(g);
}

/* ============================================================
 * 8. TIEMPO TRANSCURRIDO
 * ============================================================
 */

double elapsed_seconds(struct timespec start, struct timespec end)
{
    return (double)(end.tv_sec - start.tv_sec)
         + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}
