#WIN_20240102_16_20_59_Pro.mp4
import cv2
import mediapipe as mp
import os
import numpy as np

# Crear un objeto VideoCapture para leer el video
cap = cv2.VideoCapture('../video/parpadeoAbierto.mp4')

# Crear el objeto de detección de rostros de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Crear una carpeta para almacenar las imágenes de los rostros
output_folder = 'data/parpadeos/abiertos'
os.makedirs(output_folder, exist_ok=True)

# Contador para nombres de archivo
image_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener los resultados de la detección de rostros
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extraer las coordenadas de los puntos clave faciales
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

            # Extraer las coordenadas del rectángulo que encierra el rostro
            bboxC = cv2.boundingRect(np.array(landmarks))

            # Recortar y guardar la imagen del rostro
            face_img = frame[bboxC[1]: bboxC[1] + bboxC[3], bboxC[0]: bboxC[0] + bboxC[2]]
            cv2.imwrite(os.path.join(output_folder, f"face_{image_counter}.jpg"), face_img)
            image_counter += 1

            # Dibujar el rectángulo en el frame
            cv2.rectangle(frame, (bboxC[0], bboxC[1]), (bboxC[0] + bboxC[2], bboxC[1] + bboxC[3]), (0, 255, 0), 2)

    # Mostrar el frame con el rectángulo dibujado
    cv2.imshow('Frame', frame)

    # Romper el bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
