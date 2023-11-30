/*******************************************************************************
 *
 * vectorsum-tb.cu: vector sum with CUDA
 *
 * Programmer: Ruben Carvajal Schiaffino
 *
 * Santiago de Chile, 17/11/2023
 *
 ******************************************************************************/


#include <stdio.h>
#include <stdlib.h> 
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#define THREADxBLOCK 1024


/*
 * 
 */
void Print(unsigned int *a, unsigned int n) {
    
   unsigned int i;
   
   printf("\n");
   for (i = 0; i < n; i = i + 1)
      printf(" %d ",a[i]);
   printf("\n");
}


/*
 * 
 */
unsigned int *GenData(unsigned int *a, unsigned int n) {
 
   unsigned int i;
      
   for (i = 0; i < n; i = i + 1) 
      a[i] = (rand() % 256) + 1;
   return a;
}


/*
 * 
 */
unsigned int *AddVectorInHost(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int n) {
    
   unsigned int i;
   
   for (i = 0; i < n; i = i + 1)
      c[i] = a[i] + b[i];
   return c;
}


/*
 * 
 * Executed by the device
 * blockIdx.x: index of the element
 * 
 */
__global__ void AddVectorInDev(unsigned int *a, unsigned int *b, unsigned int *c) {

   c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}


/*
 *
 * Executed by the device
 * blockIdx.x: index of the element
 *
 */
__global__ void AddVectorInDevT(unsigned int *a, unsigned int *b, unsigned int *c) {

   c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}


/*
 *
 */
__global__ void AddVectorInDevTBe(unsigned int *a, unsigned int *b, unsigned int *c) {

   int index;

   index = threadIdx.x + blockIdx.x * blockDim.x;
   c[index] = a[index] + b[index];
}


/*
 *
 */
__global__ void AddVectorInDevTBa(unsigned int *a, unsigned int *b, unsigned int *c, unsigned int n) {

   int index;

   index = threadIdx.x + blockIdx.x * blockDim.x;
   if (index < n)
      c[index] = a[index] + b[index];
}


/*
 *
 */
int main(int argc, char **argv) {

   unsigned int *a, *b, *c;      // host copies
   unsigned int *d_a, *d_b, *d_c;  // device copies
   unsigned int n, size, mode;
   float E_cpu;
   clock_t cs, ce;
   long E_wall;
   time_t  ts, te;
   
   if (argc == 4) {
      n = atoi(argv[1]);
      mode = atoi(argv[2]);
      a = (unsigned int *) malloc(n * sizeof(unsigned int));
      b = (unsigned int *) malloc(n * sizeof(unsigned int));
      c = (unsigned int *) malloc(n * sizeof(unsigned int));
      srand(getpid());
      a = GenData(a,n);
      b = GenData(b,n);
      if (strcmp(argv[3],"-V") == 0) {
         Print(a,n);
         Print(b,n);
      }
      if (mode == 0) {
         ts = time(NULL);
         cs = clock();
         c = AddVectorInHost(a,b,c,n);
         ce = clock();
         te = time(NULL);
         E_wall = (long) (te - ts);
         E_cpu = (float)(ce - cs) / CLOCKS_PER_SEC;
         if (strcmp(argv[3],"-V") == 0)
            Print(c,n);
         printf("\n\nElapsed CPU Time: %f [Secs] Elapsed Wall Time: %ld [Secs]\n\n",E_cpu,E_wall);
      }
      else {
         size = n * sizeof(unsigned int);
         cudaMalloc((void **) &d_a,size); // allocate space for the device copies
         cudaMalloc((void **) &d_b,size);
         cudaMalloc((void **) &d_c,size);
         ts = time(NULL);
         cs = clock();
         cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice); // copy from host variable to device variable
         cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
         if (mode == 1)
            AddVectorInDev<<<n,1>>>(d_a,d_b,d_c); // Launch AddVector kernel with n blocks
         if (mode == 2)
            AddVectorInDevT<<<1,THREADxBLOCK>>>(d_a,d_b,d_c); // Launch AddVector kernel with THREADxBLOCK threads
         if (mode == 3)
            AddVectorInDevTBe<<<n / THREADxBLOCK,THREADxBLOCK>>>(d_a,d_b,d_c); // Launch AddVector kernel with n blocks and THREADxBLOCK threads
         if (mode == 4)
            AddVectorInDevTBa<<<(n + (THREADxBLOCK - 1)) / THREADxBLOCK,THREADxBLOCK>>>(d_a,d_b,d_c,n); // Launch AddVector kernel with n blocks and THREADxBLOCK threads
         cudaMemcpy(c,d_c,size,cudaMemcpyDeviceToHost); // copy from device variable to host variable
         ce = clock();
         te = time(NULL);
         E_wall = (long) (te - ts);
         E_cpu = (float)(ce - cs) / CLOCKS_PER_SEC;
         if (strcmp(argv[3],"-V") == 0)
            Print(c,n);
         printf("\n\n[CUDA] Elapsed CPU Time: %f [Secs] Elapsed Wall Time: %ld [Secs]\n\n",E_cpu,E_wall);
      }
      free(a); free(b); free(c);                    // cleanup host variables
      if (mode == 1 || mode == 2) {
         cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);  // cleanup device variables
      }
   }
   return 0;
}   
