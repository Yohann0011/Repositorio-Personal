#Librerias importadas
import cv2
import numpy as np 
from filtrokalman import filtrokalman
#Inicializacion del filtro kalman
fk = filtrokalman()
#Realizar videocaptura
cap = cv2.VideoCapture(0)
cap.set(3, 1440)
cap.set(4, 810)
#Creamos bucle While True
while True:
    #Declaracion de variables para la camara
    ret, frame = cap.read()
    #Verificamos el video
    if ret is False:
        print("No hay camara")
        break
    #Preprocesamiento de la zona de interes
    nB = np.matrix(frame[:,:,0]) #Azul
    nG = np.matrix(frame[:,:,1]) #Verde
    nR = np.matrix(frame[:,:,2]) #Rojo
    #Color
    Color = cv2.absdiff(nG,nB)
    #Binarizamos la imagen
    _, umbral = cv2.threshold(Color, 50, 255, cv2.THRESH_BINARY)
    # Extraccion de contornos de la zona seleccionada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Ordenamos los contornos del mas grande al mas pequeño
    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)
    # Dibujar los contornos extraidos
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 500:
            #Deteccion de la placa
            x, y, ancho, alto = cv2.boundingRect(contorno)
            #Extraemos las coordenadas
            xi = x #Coordenada inicial en x
            yi = y #Coordenada inicial en y
            xf = x + ancho #Coordenada final en x
            yf = y + alto #Coordenada final en y
            #Dibujo del rectangulo
            cv2.rectangle(frame,(xi,yi),(xf,yf),(255,0,0),2)
            #Dibujo del punto central
            cx = int((xi+xf)/2)
            cy = int((yi+yf)/2)
            cv2.circle(frame,(cx,cy),10,(0,0,255),-1)
            #Prediccion de la trayectoria
            predict = fk.prediccion(cx,cy)
            predict2 = fk.prediccion(predict[0],predict[1])
            predict3 = fk.prediccion(predict2[0],predict2[1])
            predict4 = fk.prediccion(predict3[0],predict3[1])
            predict5 = fk.prediccion(predict4[0],predict4[1])
        
            #Dibujamos trayectoria
            cv2.circle(frame,(predict[0], predict[1]),10,(255,0,0),-1)
            cv2.circle(frame,(predict2[0], predict2[1]),9,(255,0,0),-1)
            cv2.circle(frame,(predict3[0], predict3[1]),8,(255,0,0),-1)
            cv2.circle(frame,(predict4[0], predict4[1]),7,(255,0,0),-1)


    #Mostramos el video
    cv2.imshow("Trayectoria de objeto", frame)
    cv2.imshow("Mascara",umbral)
    #Cerrar bucle
    bandera = cv2.waitKey(1)
    if bandera == 27:
        break
cap.release()
cv2.destroyAllWindows()