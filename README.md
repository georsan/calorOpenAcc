# calorOpenAcc
# que es?
OpenACC es una especificación de programación de alto nivel que permite a los desarrolladores acelerar sus aplicaciones de cómputo científico y técnico en dispositivos de procesamiento paralelo, como GPUs, utilizando un conjunto de directivas (directives) en lugar de escribir código específico de la plataforma

# carateristicas
Algunas características que caracterizan a OpenACC son:

-s una especificación de programación de alto nivel que permite a los desarrolladores acelerar sus aplicaciones en dispositivos de procesamiento paralelo como GPUs utilizando directivas.

- No requiere escribir código específico de la plataforma como CUDA o OpenCL, lo que permite seguir utilizando lenguajes de programación de alto nivel como C, C++ o Fortran.

- Proporciona un conjunto de funciones para facilitar la programación paralela, como la creación de datos compartidos y regiones de código paralelas.

- Permite controlar cómo se manejan los datos y cómo se ejecutan las tareas en la GPU.

- Permite a los desarrolladores aprovechar el poder de las GPU para acelerar sus aplicaciones de manera fácil y eficiente.


# Diferencias con mpi
OpenACC se utiliza para acelerar aplicaciones en un solo dispositivo, mientras que MPI se utiliza para distribuir y coordinar tareas entre varios nodos en un sistema de clúster.

# Ejecicion de codigo
se ejecuta el codigo en un pc sin utilizar openacc para comparar el tiempo que se demora en hacer el calculo en los dos entornos y poder evidenciar el uso de la herramienta.

![calor sin openacc](https://github.com/georsan/calorOpenAcc/blob/de84abe0c6378d835baf822040616c2bb970c4e4/codigoLocal.png)

para correr este codigo se puede utilizar los comandos


`
gcc ecuacionCalor.c -o ecuacioCalor
`

`
./ecuacioCalor
`

para correr este codigo en una gpu se utilza 

`
gcc ecuacionCalor.c -o ecuacioCalor
`

`
./ecuacioCalor
`
