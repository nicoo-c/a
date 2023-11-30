#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void prod_tens_bloques(float *a, float *b, float *c, unsigned int k, unsigned int n){

	unsigned int indice_a = ((blockIdx.x / n) % k) + ((blockIdx.x / (k * k * n)) * k);
	unsigned int indice_b = (blockIdx.x % n) + ((((blockIdx.x / (k * n)) * n))) % (k * n);

	c[blockIdx.x] = a[indice_a] * b[indice_b];

}

__global__ void prod_tens_bloques_hilos(float *a, float *b, float *c, unsigned int m, unsigned int k, unsigned int n){

	unsigned int indice_c = threadIdx.x + (blockDim.x * blockIdx.x);
	unsigned int indice_a = ((indice_c / n) % k) + ((indice_c / (k * k * n)) * k);
	unsigned int indice_b = (indice_c % n) + ((((indice_c / (k * n)) * n))) % (k * n);
	
	if(indice_c < (m * k * k * n))
		c[indice_c] = a[indice_a] * b[indice_b];

}


int main(int argc, char *argv[]){

	unsigned int m, k, n, t_a, t_b, t_c, size_a, size_b, size_c, maxhilos;
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  int i = 0, j = 0, devCount;
  cudaDeviceProp devProp;

	float E_cpu;
  clock_t cs, ce;
  long E_wall;
  time_t  ts, te;

	if(argc == 3){

		cudaGetDeviceCount(&devCount);
		cudaGetDeviceProperties(&devProp, i);
		scanf("%i", &m);
		scanf("%i", &k);
		scanf("%i", &n);

		t_a = m * k;
		t_b = k * n;
		t_c = m * k * k * n;

		size_a = t_a * sizeof(float);
		size_b = t_b * sizeof(float);
		size_c = t_c * sizeof(float);

		maxhilos = devProp.maxThreadsPerBlock;

		a = (float *)malloc(t_a * sizeof(float));
		b = (float *)malloc(t_b * sizeof(float));
		c = (float *)malloc(t_c * sizeof(float));

		for (i = 0; i < t_a; i = i + 1)

			scanf("%f", &a[i]);

		for(i = 0; i < t_b; i = i + 1)

			scanf("%f", &b[i]);

		if(argv[2][1] ==  'V'){

			printf("\nm: %i\tk: %i\tn: %i\n", m, k, n);

			printf("\nMatriz A:\n");

			for (i = 0; i < m; i = i + 1){
				
				for(j = 0; j < k; j = j + 1)
				
						printf("%f ", a[(i * k) + j]);

				printf("\n");

			}
			
			printf("\nMatriz B:\n");

			for (i = 0; i < k; i = i + 1){
				
				for(j = 0; j < n; j = j + 1)
				
						printf("%f ", b[(i * n) + j]);

				printf("\n");

			}

			printf("\n");

		}

		cudaMalloc((void **)&d_a, size_a);
		cudaMalloc((void **)&d_b, size_b);
		cudaMalloc((void **)&d_c, size_c);

		cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

		ts = time(NULL);
		cs = clock();

		if(argv[1][1] == 'B')

			prod_tens_bloques<<<t_c, 1>>>(d_a, d_b, d_c, k, n);

		else if(argv[1][1] == 'T')

			prod_tens_bloques_hilos<<<(t_c / maxhilos) + 1, maxhilos>>>(d_a, d_b, d_c, m, k, n);

		ce = clock();
		te = time(NULL);

		E_wall = (long) (te - ts);
		E_cpu = (float) (ce - cs) / CLOCKS_PER_SEC;

		cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

		printf("Matriz C: \n");

		for(i = 0; i < m * k; i = i + 1){

			for (j = 0; j < k * n; j = j + 1)

				printf("%f ", c[i * (k * n) + j]);

			printf("\n");
		}

		if(strcmp(argv[1], "-B") == 0)

			printf("\n\nProcesamiento con %d bloques y %d hilo por bloque...\n\n Tiempo de CPU: %f segundos\tWall Time: %ld segundos\n\n", t_c, 1, E_cpu, E_wall);

		else if(strcmp(argv[1], "-T") == 0)

			printf("\n\nProcesamiento con %d bloques y %d hilos por bloque...\n\n Tiempo de CPU: %f segundos\tWall Time: %ld segundos\n\n", (t_c / maxhilos) + 1, maxhilos, E_cpu, E_wall);

		free(a);
		free(b);
		free(c);

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

	}else

		printf("\nModo de ejecución: %s -M -O < datos.txt\n\tM = {B: procesamiento solo con bloques, T: procesamiento con bloques y hebras}\n\tO = {V: modo verboso, S: modo silencioso}\n\n", argv[0]);

  return 0;
}

