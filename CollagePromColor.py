# -----------------------------------------------
# Importamos las librerías necesarias
# -----------------------------------------------
import cv2                      # OpenCV para procesamiento de imágenes
import numpy as np              # Numpy para cálculos numéricos
import os                       # OS para manejar archivos y carpetas
import matplotlib.pyplot as plt # Matplotlib para graficar el uso de los tiles

# -----------------------------------------------
# Función para calcular el color promedio de una imagen
# -----------------------------------------------
def color_promedio(img):
    img_resized = cv2.resize(img, (10, 10))     # Redimensionamos para simplificar el cálculo
    return img_resized.mean(axis=(0, 1))        # Promedio por canal (BGR)

# -----------------------------------------------
# Función para cargar los tiles y calcular sus colores promedio
# -----------------------------------------------
def cargar_tiles(ruta_tiles):
    tiles = []           # Lista para almacenar imágenes
    colores = []         # Lista para almacenar colores promedio
    for archivo in os.listdir(ruta_tiles):      # Recorremos los archivos en la carpeta
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            img = cv2.imread(os.path.join(ruta_tiles, archivo))  # Leemos la imagen
            if img is not None:
                tiles.append(img)                              # Guardamos la imagen
                colores.append(color_promedio(img))            # Calculamos y guardamos su color promedio
    return tiles, colores

# -----------------------------------------------
# Función para dividir la imagen principal en bloques
# -----------------------------------------------
def dividir_imagen(img, bloques_x, bloques_y):
    alto, ancho, _ = img.shape
    bloque_ancho = ancho // bloques_x
    bloque_alto = alto // bloques_y
    bloques = []
    for y in range(bloques_y):
        for x in range(bloques_x):
            bloque = img[y*bloque_alto:(y+1)*bloque_alto, x*bloque_ancho:(x+1)*bloque_ancho]
            bloques.append(bloque)
    return bloques, bloque_ancho, bloque_alto

# -----------------------------------------------
# Función para encontrar el tile más parecido por color promedio
# -----------------------------------------------
def encontrar_tile_mas_parecido(color_bloque, colores_tiles):
    diferencias = [np.linalg.norm(color_bloque - c) for c in colores_tiles]  # Distancia euclidiana
    return np.argmin(diferencias)  # Retorna el índice del tile más parecido

# -----------------------------------------------
# Lista global para registrar los índices de tiles usados
# -----------------------------------------------
indices_usados = []

# -----------------------------------------------
# Función para construir el mosaico final
# -----------------------------------------------
def construir_mosaico(bloques, colores_tiles, tiles, bloque_ancho, bloque_alto, bloques_x, bloques_y):
    mosaico = np.zeros((bloques_y * bloque_alto, bloques_x * bloque_ancho, 3), dtype=np.uint8)
    for i, bloque in enumerate(bloques):
        color_bloque = color_promedio(bloque)  # Calculamos el color promedio del bloque
        indice = encontrar_tile_mas_parecido(color_bloque, colores_tiles)  # Tile más parecido
        indices_usados.append(indice)  # Registramos el índice usado
        tile = cv2.resize(tiles[indice], (bloque_ancho, bloque_alto))  # Redimensionamos el tile
        y = (i // bloques_x) * bloque_alto
        x = (i % bloques_x) * bloque_ancho
        mosaico[y:y+bloque_alto, x:x+bloque_ancho] = tile  # Pegamos el tile en el mosaico
    return mosaico

# -----------------------------------------------
# Función para graficar cuántas veces se usó cada tile
# -----------------------------------------------
def graficar_uso_tiles(indices_usados):
    conteo = np.bincount(indices_usados)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(conteo)), conteo, color='cornflowerblue')
    plt.xlabel('Índice de imagen (tile)')
    plt.ylabel('Número de veces usado')
    plt.title('Frecuencia de uso de cada imagen en el mosaico')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafico_uso_tiles_color_promedio.png')
    plt.show()

# -----------------------------------------------
# BLOQUE PRINCIPAL DEL PROGRAMA
# -----------------------------------------------

# Parámetros del mosaico
bloques_x = 50
bloques_y = 50

# Cargamos la imagen principal
imagen_objetivo = cv2.imread('Monta.jpg')
if imagen_objetivo is None:
    raise Exception("No se pudo cargar la imagen objetivo.")

# Cargamos los tiles y sus colores promedio
tiles, colores_tiles = cargar_tiles('tiles')

# Dividimos la imagen en bloques
bloques, bloque_ancho, bloque_alto = dividir_imagen(imagen_objetivo, bloques_x, bloques_y)

# Construimos el mosaico
mosaico = construir_mosaico(bloques, colores_tiles, tiles, bloque_ancho, bloque_alto, bloques_x, bloques_y)

# Guardamos el mosaico como imagen
cv2.imwrite('fotomosaico_color_promedio.jpg', mosaico)

# Mostramos el mosaico en pantalla
cv2.imshow('Fotomosaico con Color Promedio', mosaico)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Generamos y mostramos el gráfico de uso de tiles
graficar_uso_tiles(indices_usados)