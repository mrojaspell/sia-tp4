# Trabajo Práctico 4 - SIA

## Requisitos:
1) Python 3
2) Pip 3


## Ejecución:
1) Ejecutar en una linea de comandos (con pip instalado):
```
pip install -r requirements.txt
```
2) Dependiendo de que ejercicio se quiere ejecutar, debe entrar en el directorio correspondiente y alterar el config.json correspondiente (se describe la configuración abajo).
3) Para cada ejercicio, ejecutar el main.py correspondiente. Ejemplo:
```
python ./main_ej1.2.py
```

## Configuración Ejercicio 1
| Campo                  | Descripción                                                                     | Valores aceptados                  |  
|------------------------|---------------------------------------------------------------------------------|------------------------------------|
| k                      | Dimensión de la grilla                                                          | Numero entero mayor a 0.           |
| radius                 | Radio de la grilla                                                              | Numero entero mayor a 0.           |
| variable_radius        | Dicta si se varía el valor del radio a lo largo de las iteraciones.             | True/False                         |
| initial_learning_rate  | Valor inicial de la constante de aprendizaje.                                   | Número punto flotante entre 0 y 1. |
| variable_learning_rate | Dicta si varía la constante de aprendizaje con las iteraciones.                 | True/False.                        |
| initialize_random      | Dicta si los pesos de la red se inicializan con valores aleatorios.             | True/False.                        |
| train_limit            | Número máximo de veces que se entrenara la red.                                 | Numero entero mayor a 0.           |
| generate_heatmap       | Dicta si se generarán o no los gráficos de calor para los paises analizados.    | True/False.                        |
| generate_U_matrix      | Dicta si se generarán o no los gráficos de matriz-U para los paises analizados. | True/False.                        |


## Configuración Ejercicio 1.2
| Campo             | Descripción                                         | Valores aceptados                  |  
|-------------------|-----------------------------------------------------|------------------------------------|
| learning_constant | Valor de la constante de aprendizaje para el método | Numero punto flotante entre 0 y 1. |
| limit             | Número máximo de veces que se entrenara la red.     | Número entero mayor a 0.           |


## Configuración Ejercicio 2
| Campo                 | Descripción                                                                      | Valores aceptados                  |  
|-----------------------|----------------------------------------------------------------------------------|------------------------------------|
| training_data         | String con la dirección relativa donde esta ubicado el archivo de entrenamiento. | Cadena de carácteres.              |
| selection_index       | Índice de patrón que se intenta reconocer.                                       | Numero entero mayor 0.             | 
| selection_size        | Cantidad de datos almacenados                                                    | Numero entero mayor 0.             | 
| noise_level           | Dicta cuanto ruido se insertará a la hora de entrenar.                           | Numero de punto flotante mayor 0.  | 
| limit                 | Cantidad máxima de veces que se entrenara la red.                                | Numero entero mayor 0.             | 
| generate_energy_graph | Dicta si se generarán los gráficos de energía para el conjunto de datos.         | True/False                         | 



