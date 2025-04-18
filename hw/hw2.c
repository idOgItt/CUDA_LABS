#include <stdio.h>
#include <stdlib.h>

int main() {
    int n;
    if (scanf("%d", &n) != 1 || n <= 0) {
        return 0;
    }

    float* arr = (float*)malloc(n * sizeof(float));
    if (!arr) {
        return 0;
    }

    for (int i = 0; i < n; i++) {
        scanf("%f", &arr[i]);
    }

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        printf("%.6e", arr[i]);
        if (i < n - 1) printf(" ");
    }
    printf("\n");

    free(arr);
    return 0;
}
