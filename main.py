import cv2

# Cargar el clasificador de rostros frontales
cara_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Activar la cámara
cam = cv2.VideoCapture(0)

# Bucle infinito para leer fotogramas de la cámara
while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convertir el fotograma a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    caras = cara_cascade.detectMultiScale(gris, 1.1, 4)

    # Dibujar rectángulos alrededor de cada rostro detectado
    for (x, y, w, h) in caras:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar la imagen con los rostros detectados
    cv2.imshow("Deteccion de Rostros", frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cam.release()
cv2.destroyAllWindows()