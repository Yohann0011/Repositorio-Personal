from tkinter import *
from tkinter import filedialog, simpledialog, messagebox
from turtle import bgcolor
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import os
from pickletools import optimize
import numpy as np
import pathlib
import mediapipe as mp
from time import sleep

os.system("cls")
print('Cargando...')
dataLocal = os.path.dirname(os.path.abspath(__file__))
dataPath = dataLocal + '/Personas'
imagePath = os.listdir(dataPath)
global count
count = 0
global faceClassif
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#llamamos a mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def modelo():
    global face_recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(dataLocal + '/' + 'Mod_Front_Cara_2022.xml')

def iniciar():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3,1920)
    cap.set(4,1080)
    visualizar()

def iniciarG():
    global cap
    global personPath
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3,1920)
    cap.set(4,1080)
    #personName = nombre del rostro
    Name= simpledialog.askstring("Nombre","Ingresa el nombre")
    #print(Name)
    if Name == '':
        messagebox.showinfo(message="El nombre no puede estar vacío", title="Alerta")
    elif Name == "None":
        menu()
    else:
        personName = str(Name)
        personPath = dataPath + '/' + personName   
        vistaA()

def iniciarGV():
    global cap
    global personPath
    path_video = filedialog.askopenfilename(filetypes = [
            ("all video format", ".mp4"),
            ("all video format", ".avi")])
    if len(path_video) > 0:
        btnAgregarV.configure(state="disable")
    global cap
    cap = cv2.VideoCapture(path_video)
    cap.set(3,1920)
    cap.set(4,1080)
    #personName = nombre del rostro
    Name= simpledialog.askstring("Nombre","Ingresa el nombre")
    #print(Name)
    if Name == '':
        messagebox.showinfo(message="El nombre no puede estar vacío", title="Alerta")
    elif Name == "None":
        menu()
    else:
        personName = str(Name)
        personPath = dataPath + '/' + personName   
        vistaA()

def selec_video():
    path_video = filedialog.askopenfilename(filetypes = [
            ("all video format", ".mp4"),
            ("all video format", ".avi")])
    if len(path_video) > 0:
        btnAgregarV.configure(state="disable")
    global cap
    cap = cv2.VideoCapture(path_video)
    cap.set(3,1920)
    cap.set(4,1080)
    visualizar()

def grabarRostro(Out):
    global count
    #os.system("cls")
    #llamamos a mediapipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    #llamamos las configuraciones
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

    #Creamos bucle mientras para ejecucion
        while True:
            
            ret, frame = cap.read()
            if ret == False:
                if count >=1 or count <=499:
                    messagebox.showinfo(message="Tomas insuficientes \nProceso Finalizado", title="Alerta")
                    finalizar()
                    #print('Tomas insuficientes \nProceso Finalizado')
                else:
                    messagebox.showinfo(message="No se inicio la captura de imagen", title="Alerta")
                    finalizar()
                    #print('No se inicio la captura de imagen')
                break
            ############## CREACION DE LA MASCARA DEL FONDO ##########################
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(frame_rgb)
            _, th = cv2.threshold(results.segmentation_mask, 0.65 , 255 , cv2.THRESH_BINARY)
            #print(th.dtype)
            th = th.astype(np.uint8)
            #Suavisado de bordes
            th = cv2.medianBlur(th, 35, 1)
            th_inv = cv2.bitwise_not(th)

            #Fondo con color
            #bg_image = np.ones(frame.shape, dtype=np.uint8)
            #bg_image[:] = BG_COLOR
            #Fondo Difulminado (x,y) orientacion del difulminado
            ksize = (9, 9)
            #bg_image = cv2.blur(frame, ksize)
            bg_image = cv2.GaussianBlur(frame, ksize, 5)
            
            #Uniendo Fondo y Mascara
            bg = cv2.bitwise_and(bg_image,bg_image, mask = th_inv)
            #Uniendo Inverso y Segmentado
            fg = cv2.bitwise_and(frame, frame, mask = th)
            #Union de Capas
            Out = cv2.add(bg , fg)

            ################## TOMA  DE FOTOGRAFIAS Y DETECCION DE ROSTRO #######################

            #Resolucion de la ventana
            Out = imutils.resize(Out, width=1440, height=800)
            gray = cv2.cvtColor(Out,cv2.COLOR_RGBA2GRAY)
            auxFrame = Out.copy()  

            faces = faceClassif.detectMultiScale( gray, scaleFactor=1.10,  minNeighbors=15,  minSize=(200,200),  maxSize=(700,700))
            
            imgAux = personPath + '/r_{}.jpg'

            for(x, y, w, h) in faces:
                cv2.rectangle(Out, (x,y),(x+w, y+h),(255,0,0),2)
                rostro = auxFrame[y:y + h, x:x + w]
                #Medida en px de la imagen capturada del rostro
                rostro = cv2.resize(rostro, (180,180), interpolation=cv2.INTER_CUBIC)
                if h <= 250:
                    cv2.rectangle(frame,(453,10),(907,90),(0,0,0),-1)
                    cv2.putText(frame, 'Rostro lejano', (475,70), 2, 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    #Generar Imagen - Rostro es el nombre de las fotos de las personas
                    cv2.imwrite(imgAux.format(count), rostro)
                    #optimizar el tamaño de la imagen creada
                    img = Image.open(imgAux.format(count))
                    img.save(personPath + '/f_{}.jpg'.format(count), optimize = True, quality=75)
                    #remover la imagen generada al inicio
                    os.remove(imgAux.format(count))
                
                rostro = cv2.resize(rostro, (800,800), interpolation=cv2.INTER_CUBIC)
                Out = rostro
                count = count+1
                #print (count)
                if count >= 300:
                    print("Fotos obtenidas")
                    Entreno()
            return Out

def visualizar():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            btnFinalizar.configure(state="normal")
            btnAgregarV.configure(state="disable")
            btnAgregar.configure(state="disable")
            btnIniciar.configure(state="disable")
            btnIniciarV.configure(state="disable")
            btnEntrenar.configure(state="disable")
            frame = imutils.resize(frame, width=1200)
            frame = deteccion_facilal(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img,width=1300, height=750)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
        else:
            lblVideo.image = ""
            cap.release()

def vistaA():
        #########################################

    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)

    #########################################
    global cap
    if cap is not None:
        ret, Out = cap.read()
        if ret == True:
            btnFinalizar.configure(state="normal")
            btnAgregarV.configure(state="disable")
            btnAgregar.configure(state="disable")
            btnIniciar.configure(state="disable")
            btnIniciarV.configure(state="disable")
            btnEntrenar.configure(state="disable")
            Out = imutils.resize(Out, width=1440, height=800)
            Out = grabarRostro(Out)
            Out = cv2.cvtColor(Out, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(Out)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img,width=800,height=800)#width=1440,height=800
            lblVideo.image = img
            lblVideo.after(10, vistaA)
        else:
            lblVideo.image = ""
            cap.release()

def deteccion_facilal(frame):
    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame = imutils.resize(frame,width=1440,height=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = faceClassif.detectMultiScale(gray, scaleFactor=1.3,  minNeighbors=8,  minSize=(120,120),  maxSize=(650,650))
        #print(faces)
        for(x, y, w, h) in faces:
            #Regtangulo sobre el rostro
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            #Medida en px de la imagen capturada del rostro
            rostro = cv2.resize(rostro, (180,180), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            #Etiqueta de identificador con valores
            #cv2.putText(frame, '{}'.format(result), (x, y), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
            if h <= 179:
                    cv2.rectangle(frame,(453,10),(907,90),(0,0,0),-1)
                    cv2.putText(frame, 'Rostro lejano', (475,70), 2, 2, (255, 255, 255), 2, cv2.LINE_AA)
            alt = x+w/3

            if result[1]<60:
                #Nombre o etiqueta del rostro
                cv2.putText(frame, '{}'.format(imagePath[result[0]]), (int(alt), y+h+50), 2, 1.1, (36, 229, 130), 2, cv2.LINE_AA)
                #Cuadro identificador de rostro
                #cv2.rectangle(frame, (x, y),(x+w, y+h),(0,255,0),1)
            else:
                #Nombre o etiqueta del rostro
                cv2.putText(frame, 'Desconocido...', (int(alt), y+h+50), 2, 0.8, (44, 44, 225), 2, cv2.LINE_AA)
                #Cuadro identificador de rostro
                #cv2.rectangle(frame, (x, y),(x+w, y+h),(0,0,255),1)
        return frame

def Entreno():
    os.system("cls")
    btnFinalizar.configure(state="disable")
    btnAgregarV.configure(state="disable")
    btnAgregar.configure(state="disable")
    btnIniciar.configure(state="disable")
    btnIniciarV.configure(state="disable")
    btnEntrenar.configure(state="disable")
    ListaGente = os.listdir(dataPath)
    #print('Lista de personas:', ListaGente)

    caras = []
    datosRostros = []
    ContCaras = 0
    count = 1

    for nameDir in ListaGente:
        personPath = dataPath + '/' + nameDir
        #print('Leyendo Imgs carpeta {}'.format(count))
        count = count+1
        for nombArc in os.listdir(personPath):
            #print('Rostro :', nameDir + '/' + nombArc)
            caras.append(ContCaras)

            datosRostros.append(cv2.imread(personPath + '/' + nombArc, 0))
            image = cv2.imread(personPath + '/' + nombArc, 0)

            #Mostrar el funcionamiento del conteo de imagenes
            #cv2.imshow('Lectura de Imagen',image)
            #cv2.waitKey(10)
            #################################################
        
        ContCaras = ContCaras + 1

    #cv2.destroyAllWindows()

    #Contar la cantidad de etiquetas
    #print('Rostros',caras)
    #print('Numero de etiquetas 0: ',np.count_nonzero(np.array(caras)==0))
    #print('Numero de etiquetas 0: ',np.count_nonzero(np.array(caras)==1))

    # Método para entrenar el reconocedor
    face_recognizer
    messagebox.showinfo(message="Entrenando...\nEsto pude tomar un tiempo", title="Entrenamiento",)
    #print('Entrenando...')
    import time
    inicio = time.time()    
    face_recognizer.train(datosRostros, np.array(caras))
    face_recognizer.write(dataLocal + '/' + 'Mod_Front_Cara_2022.xml')
    #print("Modelo Guardado")
    modelo()
    tiempoEntrenamiento = time.time()-inicio
    print(tiempoEntrenamiento)
    messagebox.showinfo(message="Modelo Guardado", title="Entrenamiento")
    menu()

import sys
                                                                    
def progressbar(it, prefix="", size=60, file=sys.stdout):
    os.system("cls")
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
        file.write("\n")
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
        file.write("\n")
    file.flush()


def finalizar():
    global cap
    cap.release()
    lblVideo.config(width=1,height=100)
    menu()

def menu():
    os.system("cls")
    print('En Ejecución')
    count = 0
    global btnAgregar
    global btnAgregarV
    global btnEntrenar
    global btnIniciarV
    global btnIniciar
    global btnFinalizar
    global lblVideo

    root.title("I SEE YOU")
    root.resizable(0,0)
    root.configure(bg="#002c43")
    #Poner imagen de fondo
    dataLocal = os.path.dirname(os.path.abspath(__file__)) +'/fondo.ppm'
    fon = dataLocal
    Img_Fondo_main = PhotoImage(file = fon)
    lbl_Fondo = Label(root, image=Img_Fondo_main, bd=0)
    lbl_Fondo.place(x=-50,y=0)
    lblSpace1 = Label(root, bg="#0d2135")
    lblSpace1.grid(column=0, row=0, padx=30, pady=20)
    #Label del titulo
    lbl_titulo = Label(root, text = "BIENVENIDO", fg="white", font=("Eras Demi ITC", 24), bg="#00366B")
    lbl_titulo.grid(column=0, row=1, padx=30, pady=10)
    btnAgregar = Button(root, text="Agregar Rostro Stream", width=30, command=iniciarG, fg="black", font="Verdana 14", bg="light blue", cursor = "hand2", relief="flat", overrelief="flat")
    btnAgregar.grid(column=0, row=2, padx=30, pady=10)
    btnAgregarV = Button(root, text="Agregar Rostro Video", width=30, command=iniciarGV, fg="black", font="Verdana 14", bg="light blue", cursor = "hand2", relief="flat", overrelief="flat")
    btnAgregarV.grid(column=0, row=3, padx=30, pady=10)
    btnEntrenar = Button(root, text="Entrenar", width=30, command=Entreno, fg="black", font="Verdana 14", bg="light blue", cursor = "hand2", relief="flat", overrelief="flat")
    btnEntrenar.grid(column=0, row=4, padx=30, pady=10)
    btnIniciarV = Button(root, text="Iniciar Video", width=30, command=selec_video, fg="black", font="Verdana 14", bg="light blue", cursor = "hand2", relief="flat", overrelief="flat")
    btnIniciarV.grid(column=0, row=5, padx=30, pady=10)
    btnIniciar = Button(root, text="Iniciar Stream", width=30, command=iniciar, fg="black", font="Verdana 14", bg="light blue", cursor = "hand2", relief="flat", overrelief="flat")
    btnIniciar.grid(column=0, row=6, padx=30, pady=10)
    btnFinalizar = Button(root, text="Finalizar", width=30, command=finalizar, fg="black", font="Verdana 14", bg="light blue", cursor = "hand2", relief="flat", overrelief="flat")
    btnFinalizar.grid(column=0, row=7, padx=30, pady=10)
    btnFinalizar.configure(state="disable")
    lblSpace2 = Label(root, bg="#142d4b")
    lblSpace2.grid(column=0, row=8, padx=30, pady=20)
    lblVideo = Label(root, bg="#002c43")
    lblVideo.grid(column=1, row=0, rowspan=90)
    

    btnIniciar.configure(state="normal")
    btnIniciarV.configure(state="normal")
    btnEntrenar.configure(state="normal")
    btnAgregar.configure(state="normal")
    btnAgregarV.configure(state="normal")

    root.mainloop()

def inicio():
    modelo()
    menu()
cap = None
root = Tk()
inicio()