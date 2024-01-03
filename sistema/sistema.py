import cv2
import numpy as np
from keras.models import load_model

# Cargar el modelo preentrenado
modelo_parpadeos = load_model('../modelo_parpadeos.h5')

# Función para realizar la detección de parpadeos
def detectar_parpadeos(frame, umbral=0.5):
    # Redimensionar la imagen para que coincida con las expectativas del modelo
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    frame = np.reshape(frame, (1, 64, 64, 3))

    # Realizar la predicción
    resultado = modelo_parpadeos.predict(frame)

    # Comparar la probabilidad de parpadeo con el umbral
    if resultado[0][0] > umbral:
        return True
    else:
        return False

# Inicializar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    # Realizar la detección de parpadeos en el fotograma actual
    parpadeo_detectado = detectar_parpadeos(frame)

    # Mostrar el resultado en el fotograma
    if parpadeo_detectado:
        cv2.putText(frame, "Parpadeo detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Ojos abiertos", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar el fotograma en una ventana
    cv2.imshow("Detección de Parpadeos", frame)

    # Salir con la tecla 'Esc'
    if cv2.waitKey(20) & 0xFF == 27:
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
