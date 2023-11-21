# se calcula ciertas distancias con la fórmula, ya que nos ayudará a calcular la relación expecto del ojo
# primero se calculará la distancia de del punto 2 al punto 5 distancia a
# distancia del punto 3 al punto 5 distancia b
# punto 1 al punto 4 distancia c
# todas estas distancias nos permitira calcular la relación especto del ojo

# importar los módulos de cv2 y mediapipe y numpy
import cv2
import mediapipe as mp
import numpy as np

from time import time
from turtle import width
import time


# función drawing_output
# frame son los fotogramas
# pedir las coordenas del ojo izquierdo y derecho
# blink_counter contador de parpadeos
def drawing_output(frame, coordinates_left_eye, corrdinates_right_eye, blink_counter):
    # efecto de la transparencia sobre el area de ambos ojos
    # azul y negro imagen
    aux_image = np.zeros(frame.shape, np.uint8)
    # contornos del ojo izquierdo
    contours1 = np.array([coordinates_left_eye])
    # contornos del ojo derecho
    contours2 = np.array([coordinates_right_eye])
    # para poder dibujar estos contornos en aux_image coor azul
    cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
    cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))
    # visualizar la imagen que se tiene
    # para sumar dos imágenes con transparencia en frame 1 y aux_image de 0.7 1
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)
    # para mostrar el numparpadeos
    cv2.putText(output, "Number of blinks:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # para mostrar el número de parpadeos
    cv2.putText(output, "{}".format(blink_counter), (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)
    # acá termina para mostrar el número de parpadeos
    return output  # acá termina la función drawing_output


# función que resive las seis coodenadas de cada ojo
# aplicar la distancia euclidiana para obtener las dos distancias verticales y la distancia horizontal
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    # return el aspecto del ojo
    return (d_A + d_B) / (2 * d_C)


# ------
def distancia_vertical(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    # return el aspecto del ojo
    return d_A


# ------************************************************************************************************************************

# cero para especificar la webcam
# indicar el video que se está leyendo

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# se utiliza mediapipe face mesh
# utilizar llamando y almacenando en la variable

mp_face_mesh = mp.solutions.face_mesh

# acceder a los seis puntos por cada ojo
# estos ciertos puntos nos permitiran calcular dos distancias verticales y una distancia horizontal
# hay que tomar en cuenta que el primer punto es el de la ezquina izquiera y luego tomaremos los puntos en sentido horario

# índices del ojo izquierdo
index_left_eye = [33, 160, 158, 133, 153, 144]

# índices ojo derecho

index_right_eye = [362, 385, 387, 263, 373, 380]

# umbral para ver los ojos cerrados o abiertos
# declarando una constante
# en base a la experimentación
# experimentación #umbral

EAR_THRESH = 0.26
# variables

aux_counter = 0
parpadeo = False
tiempo = 0
inicio = 0
final = 0
muestra = 0
conteo_sue = 0

# termino de variables

# número consecutivos de frames que se presenta cuando los ojos están cerrados
# como mínimo se debe cumplir dos fotogramas
# experimentación #frames consecutivos

NUM_FRAMES = 2

# Cuenta el número de parpadeos
blink_counter = 0

# opciones de configuración
with mp_face_mesh.FaceMesh(

        # false entonces la solución trata como las imágenes de entrada como video en vivo
        # se aplica el detector de rostros en imágenes
        static_image_mode=False,
        # número máximo de rostros a detectar
        max_num_faces=1) as face_mesh:
    # imagen en rgb para que pueda realizar la detección mediapipe
    # leer los fotogramas del video
    while True:
        ret, frame = cap.read()  # captura de la video captura
        if ret == False:
            break
        # visualizar estos doce puntos en frame

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        # frame rgb
        # imagen de bgr a rgb
        # aplicar la detección con results adentro especificamos la imágen en rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # aplicar facemesh a frame
        results = face_mesh.process(frame_rgb)
        # hasta acá
        # condicional para ver si se obtiene resultados de los rostros detectados
        # unas listas vacias donde se almacerán las corrdenadas de los ojos x e y

        coordinates_left_eye = []
        coordinates_right_eye = []

        # condicional para ver si se obtiene resultados de los rostros detectados
        if results.multi_face_landmarks is not None:
            # acá se accede a todos los putnos de la malla facial
            for face_landmarks in results.multi_face_landmarks:
                # los únicos que necesitamos es izquierdo y derecho
                # este for recorre cada punto de los índices
                for index in index_left_eye:
                    # extraemos las coordenadas de dicho índices
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    # acá se almacenarán con los puntos de coordenadas x e y del ojo izquierdo con append para que se pueda guardar cada corrdenada
                    coordinates_left_eye.append([x, y])
                    # visualizar estos puntos con cvr.circle
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 250), 1)
                    # los únicos que necesitamos es izquierdo y derecho
                    # este for recorre cada punto de los índices
                for index in index_right_eye:
                    # extraemos las coordenadas de dicho índices
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    # acá se almacenarán con los puntos de coordenadas x e y del ojo derecho con append para que se pueda guardar cada corrdenada
                    coordinates_right_eye.append([x, y])
                    # visualizar estos puntos con cvr.circle
                    cv2.circle(frame, (x, y), 2, (128, 0, 250), 1)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                    # llamámos a la función eye_aspect_ratio dando las corrdenadas del ojo izquierdo y ojo derecho almacenando a una variable
            ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
            ear_right_eye = eye_aspect_ratio(coordinates_right_eye)

            # sumar y divir entre dos para obtener las relaciones de aspecto para obtener el promedio de estos dos ojos
            ear = (ear_left_eye + ear_right_eye) / 2
            # ************
            # para obtener la longitud del cierre del ojo izquierdo y ojo derecho
            longitud1 = distancia_vertical(coordinates_left_eye)
            longitud2 = distancia_vertical(coordinates_right_eye)

            # ************

            cv2.putText(frame, "Micro dreams:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "{}".format(conteo_sue), (220, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)
            # --------------

            # --------------
            cv2.putText(frame, "Duration:", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "{}".format(muestra), (220, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 0, 250), 2)
            # --------------
            # mide el tiempo de microsueños después de 3 segundos o más
            if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                parpadeo = True
                inicio = time.time()
            elif longitud2 > 10 and longitud2 > 10 and parpadeo == True:
                parpadeo = False
                final = time.time()
            tiempo = round(final - inicio, 0)
            if tiempo >= 3:
                conteo_sue = conteo_sue + 1
                muestra = tiempo
                inicio = 0
                final = 0
            # *********************************************************************
            if ear < EAR_THRESH:
                # cuántas veces se cumple esta condición de fotogramas
                aux_counter += 1
                print(longitud1, "LONGITUD")
            else:
                # si los ojos están abiertos de nuevo

                if aux_counter >= NUM_FRAMES:
                    # se vuelve a cero para un segundo conteo

                    aux_counter = 0
                    # cuenta el número de parpadeos

                    blink_counter += 1

            # para visualizar la función drawing_output y para que se visualice la función
            # y para visualizar una sola ventana
            frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, blink_counter)

        cv2.imshow("Frame", frame)
        # open cv analiza las imágenes lo más antes posible
        k = cv2.waitKey(20) & 0xFF  # 20
        if k == 27:
            break
cap.release()
cv2.destroyAllWindows()