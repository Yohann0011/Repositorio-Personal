#Librerias importadas
import cv2
import numpy as np 
# Creamos la clase
class filtrokalman:
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    #funcion de prediccion
    def prediccion(self, cx, cy):
        #Estimacion de posicion
        medida = np.array([[np.float32(cx)],[np.float32(cy)]])
        self.kf.correct(medida)
        predict = self.kf.predict()
        x, y = int(predict[0]), int(predict[1])
        return x,y