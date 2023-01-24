#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <time.h>

const int Nx=500;  // Número de puntos en x
const int Ny=500;  // Número de puntos en y
const double Lx=100.0; // Tamaño de la región en x
const double Ly=100.0; // Tamaño de la región en y
const double alpha=0.01; // Conductividad térmica
const double dt=0.010; // Tamaño del paso temporal
const double Tf=100.0; // Tiempo final

/*
 esta funcion actualiza la solución de la ecuación de calor en una malla 2D utilizando derivadas. El primer bucle anidado recorre los puntos de la malla en el eje x, mientras que el segundo bucle anidado recorre los puntos de la malla en el eje y. 
*/
void updateTemperature(double u[Nx][Ny], double dx, double dy, double dt) {
    double unew[Nx][Ny];
    // inicia al compilador que ejecute el bloque en la gpu y que combine el bucle en uno solo
    #pragma acc parallel loop collapse(2)
    for (int i = 1; i < Nx - 1; i++) {
	for (int j = 1; j < Ny - 1; j++) {
	    unew[i][j] = u[i][j] + alpha * dt / (dx * dx) * (u[i+1][j] - 2 * u[i][j] + u[i-1][j]) + 
	                 alpha * dt / (dy * dy) * (u[i][j+1] - 2 * u[i][j] + u[i][j-1]);
	}
    }
    #pragma acc parallel loop collapse(2)
    for (int i = 1; i < Nx - 1; i++) {
	for (int j = 1; j < Ny - 1; j++) {
	    u[i][j] = unew[i][j];
	}
    }
}

int main() {
	
	double dx = Lx / (Nx - 1);	//derivadas en x
	double dy = Ly / (Ny - 1);	//derivadas en y
	double u[Nx][Ny];//distribución de temperatura en una región
	double t = 0.0;
	clock_t start, end;
	start = clock();

	#pragma acc parallel loop collapse(2)
	for (int i = 0; i < Nx; i++) {
	for (int j = 0; j < Ny; j++) {
	u[i][j] = 0.0;
	}
	}
	#pragma acc parallel loop collapse(2)
	for (int i = 0; i < Nx; i++) {
	u[i][0] = 1.0;
	u[i][Ny-1] = 1.0;
	}
	#pragma acc parallel loop collapse(2)
	for (int j = 0; j < Ny; j++) {
	u[0][j] = 1.0;
	u[Nx-1][j] = 1.0;
	}
	while (t < Tf) {
	updateTemperature(u, dx, dy, dt);
	t += dt;
	}
	end = clock();
	double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Tiempo de ejecución: %lf ms", time_taken * 1000);
	printf("/n");
	return 0;
}




