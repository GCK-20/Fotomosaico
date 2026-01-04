# -----------------------------------------------
# Importamos las librerías necesarias
# -----------------------------------------------
import cv2                      # OpenCV: librería para procesamiento y manipulación de imágenes
import numpy as np              # Numpy: operaciones matemáticas y manejo de arreglos
import os                       # OS: manejo de archivos y carpetas del sistema
import matplotlib.pyplot as plt # Matplotlib: para graficar estadísticas y visualizaciones

# -----------------------------------------------
# Función para calcular el color promedio de una imagen
# -----------------------------------------------
def color_promedio(img):
    img_resized = cv2.resize(img, (10, 10))     # Redimensionamos la imagen a 10x10 píxeles para simplificar el cálculo
    return img_resized.mean(axis=(0, 1))        # Calculamos el promedio de color por canal (BGR) en toda la imagen

# -----------------------------------------------
# Función para cargar los tiles y calcular sus colores promedio
# -----------------------------------------------
def cargar_tiles(ruta_tiles):
    tiles = []           # Lista donde se guardarán las imágenes de los tiles
    colores = []         # Lista donde se guardarán los colores promedio de cada tile
    for archivo in os.listdir(ruta_tiles):      # Recorremos todos los archivos dentro de la carpeta indicada
        if archivo.endswith('.jpg') or archivo.endswith('.png'):  # Filtramos solo imágenes con extensión .jpg o .png
            img = cv2.imread(os.path.join(ruta_tiles, archivo))   # Leemos la imagen con OpenCV
            if img is not None:                                   # Verificamos que la imagen se haya cargado correctamente
                tiles.append(img)                                 # Guardamos la imagen en la lista de tiles
                colores.append(color_promedio(img))               # Calculamos y guardamos su color promedio
    return tiles, colores                                         # Retornamos las listas de imágenes y colores

# -----------------------------------------------
# Función para dividir la imagen principal en bloques
# -----------------------------------------------
def dividir_imagen(img, bloques_x, bloques_y):
    alto, ancho, _ = img.shape              # Obtenemos dimensiones de la imagen (alto, ancho y canales de color)
    bloque_ancho = ancho // bloques_x       # Calculamos el ancho de cada bloque
    bloque_alto = alto // bloques_y         # Calculamos el alto de cada bloque
    bloques = []                            # Lista para almacenar los bloques de la imagen
    for y in range(bloques_y):              # Recorremos filas de bloques
        for x in range(bloques_x):          # Recorremos columnas de bloques
            bloque = img[y*bloque_alto:(y+1)*bloque_alto, x*bloque_ancho:(x+1)*bloque_ancho]  # Extraemos el bloque
            bloques.append(bloque)          # Guardamos el bloque en la lista
    return bloques, bloque_ancho, bloque_alto  # Retornamos los bloques y sus dimensiones

# -----------------------------------------------
# Función para encontrar el tile más parecido por color promedio
# -----------------------------------------------
def encontrar_tile_mas_parecido(color_bloque, colores_tiles):
    diferencias = [np.linalg.norm(color_bloque - c) for c in colores_tiles]  # Calculamos distancia euclidiana entre colores
    return np.argmin(diferencias)  # Retornamos el índice del tile con menor diferencia (más parecido)

# -----------------------------------------------
# Lista global para registrar los índices de tiles usados
# -----------------------------------------------
indices_usados = []  # Aquí se guardarán los índices de los tiles seleccionados para cada bloque

# -----------------------------------------------
# Función para construir el mosaico final
# -----------------------------------------------
def construir_mosaico(bloques, colores_tiles, tiles, bloque_ancho, bloque_alto, bloques_x, bloques_y):
    # Creamos una imagen vacía del tamaño total del mosaico
    mosaico = np.zeros((bloques_y * bloque_alto, bloques_x * bloque_ancho, 3), dtype=np.uint8)
    
    # Recorremos cada bloque de la imagen original
    for i, bloque in enumerate(bloques):
        color_bloque = color_promedio(bloque)  # Calculamos el color promedio del bloque
        indice = encontrar_tile_mas_parecido(color_bloque, colores_tiles)  # Buscamos el tile más parecido
        indices_usados.append(indice)  # Guardamos el índice del tile usado
        tile = cv2.resize(tiles[indice], (bloque_ancho, bloque_alto))  # Redimensionamos el tile al tamaño del bloque
        
        # Calculamos la posición donde debe colocarse el tile dentro del mosaico
        y = (i // bloques_x) * bloque_alto
        x = (i % bloques_x) * bloque_ancho
        
        # Pegamos el tile en la posición correspondiente
        mosaico[y:y+bloque_alto, x:x+bloque_ancho] = tile
    
    return mosaico  # Retornamos el mosaico final

# -----------------------------------------------
# Función para graficar cuántas veces se usó cada tile
# -----------------------------------------------
def graficar_uso_tiles(indices_usados):
    conteo = np.bincount(indices_usados)  # Contamos cuántas veces aparece cada índice
    plt.figure(figsize=(12, 6))           # Definimos tamaño de la gráfica
    plt.bar(range(len(conteo)), conteo, color='cornflowerblue')  # Dibujamos barras con frecuencia de uso
    plt.xlabel('Índice de imagen (tile)') # Etiqueta del eje X
    plt.ylabel('Número de veces usado')   # Etiqueta del eje Y
    plt.title('Frecuencia de uso de cada imagen en el mosaico')  # Título del gráfico
    plt.grid(True)                        # Activamos cuadrícula
    plt.tight_layout()                    # Ajustamos márgenes
    plt.savefig('grafico_uso_tiles_color_promedio.png')  # Guardamos la gráfica como imagen
    plt.show()                            # Mostramos la gráfica en pantalla

# -----------------------------------------------
# BLOQUE PRINCIPAL DEL PROGRAMA
# -----------------------------------------------

# Parámetros del mosaico: número de bloques en X y Y
bloques_x = 50
bloques_y = 50

# Cargamos la imagen principal que queremos convertir en mosaico
imagen_objetivo = cv2.imread('foto_objListo.jpg')
if imagen_objetivo is None:  # Validamos que la imagen se haya cargado correctamente
    raise Exception("No se pudo cargar la imagen objetivo.")

# Cargamos los tiles y sus colores promedio desde la carpeta 'tiles'
tiles, colores_tiles = cargar_tiles('tiles')

# Dividimos la imagen objetivo en bloques
bloques, bloque_ancho, bloque_alto = dividir_imagen(imagen_objetivo, bloques_x, bloques_y)

# Construimos el mosaico reemplazando cada bloque por el tile más parecido
mosaico = construir_mosaico(bloques, colores_tiles, tiles, bloque_ancho, bloque_alto, bloques_x, bloques_y)

# Guardamos el mosaico como archivo de imagen
cv2.imwrite('fotomosaico_color_promedio.jpg', mosaico)

# Mostramos el mosaico en pantalla
cv2.imshow('Fotomosaico con Color Promedio', mosaico)
cv2.waitKey(0)              # Esperamos a que se presione una tecla
cv2.destroyAllWindows()     # Cerramos la ventana

# Generamos y mostramos el gráfico de uso de tiles
graficar_uso_tiles(indices_usados)
