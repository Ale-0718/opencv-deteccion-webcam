import cv2

# Cargar el clasificador de rostros frontales
cara_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cargar la imagen desde el archivo "imagen.jpg"
# Asegúrate de que "imagen.jpg" esté en la misma carpeta que tu script, o proporciona la ruta completa.
imagen = cv2.imread("imagen.jpg")

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que 'imagen.jpg' exista y la ruta sea correcta.")
    exit() # Salir del programa si no se carga la imagen

# Convertir la imagen a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
# Los parámetros scaleFactor y minNeighbors pueden ajustarse para mejorar la detección.
caras = cara_cascade.detectMultiScale(gris, 1.1, 4)

# Aplicar Canny para detectar bordes
# Los parámetros 100 y 200 son los umbrales inferior y superior para el algoritmo Canny.
bordes = cv2.Canny(gris, 100, 200)

# Dibujar rectángulos alrededor de cada rostro detectado
# Se dibuja sobre la imagen original a color.
for (x, y, w, h) in caras:
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2) # Rectángulo verde

# Mostrar la imagen original con los rostros detectados
cv2.imshow("Deteccion de Rostros en Imagen", imagen)

# Mostrar los bordes detectados (Canny) en una ventana separada
cv2.imshow("Bordes Detectados (Canny) en Imagen", bordes)

# Esperar hasta que el usuario presione una tecla para cerrar las ventanas
cv2.waitKey(0)

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()