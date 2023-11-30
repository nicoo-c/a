#include <stdio.h>
#include <stdlib.h>

int main(){

  unsigned int m, k, n, t_a, t_b, t_c;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int i = 0;
    scanf("%i", &m);
    scanf("%i", &k);
    scanf("%i", &n);

    t_a = m * k;
    t_b = k * n;
    t_c = m * k * k * n;

    a = (float *)malloc(t_a * sizeof(float));
    b = (float *)malloc(t_b * sizeof(float));
    c = (float *)malloc(t_c * sizeof(float));

    printf("\nMatriz A:\n");

    for (i = 0; i < t_a; i = i + 1)
    {
        scanf("%f", &a[i]);
        printf("%f ", a[i]);
    }

    printf("\nMatriz B:\n");

    for (i = 0; i < t_b; i = i + 1)
    {
        scanf("%f", &b[i]);
        printf("%f ", b[i]);
    }

    for(i = 0; i < t_c; i = i + 1){

      printf("\nc[%d] = a[%d] * b[%d]\n", i, ((i / n) % k) + ((i / (k * k * n)) * k), (i  % n) + ((((i / (k * n)) * n))) % (k * n)); // BlockIdx.x := i
      c[i] = a[((i / n) % k) + ((i / (k * k * n)) * k)] * b[(i  % n) + ((((i / (k * n)) * n))) % (k * n)];

    }

    printf("\n");

    for(i = 0; i < m * k; i = i + 1){

      for(int j = 0; j < k * n; j = j + 1)

        printf("%f ", c[i * (k * n) + j]);

      printf("\n");

    }

  return 0;
}