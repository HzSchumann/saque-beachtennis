import numpy as np
import pickle
import cv2

filename = "renato1_1"
# cap = cv2.VideoCapture(f'serve/{filename}.mp4')
cap = cv2.VideoCapture(f'serve/{filename}.mp4')

# filename = "henzo01"
# cap = cv2.VideoCapture(f'serve/{filename}.mp4')
# cap = cv2.VideoCapture(f'{filename}.mov')

# intervalor_baixo = np.array([31, 50, 166])
intervalor_baixo = np.array([1, 70, 140])
# intervalo_alto = np.array([40, 200, 255])
intervalo_alto = np.array([10, 200, 220])

# Intervalos de cores para laranja
intervalo_baixo_laranja = np.array([1, 70, 140])
intervalo_alto_laranja = np.array([10, 200, 220])

# Intervalos de cores para verde
intervalo_baixo_verde = np.array([10, 70, 190])
intervalo_alto_verde = np.array([50, 90, 255])

#Intervalo teste
lower_color = np.array([2, 60, 168])  # Ajuste a faixa de cor conforme necessário
upper_color = np.array([14,  200, 202])

list_balls = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hsv
    #h -> 0 - 365 / 0 - 180
    # s e v -> 0 - 100 / 0 - 255
    # mascara = cv2.inRange(hsv, intervalor_baixo, intervalo_alto)

    # mascara = cv2.inRange(hsv, intervalo_baixo_verde, intervalo_alto_verde)
    mascara = cv2.inRange(hsv, lower_color, upper_color)

    mascara_laranja = cv2.inRange(hsv, intervalo_baixo_laranja, intervalo_alto_laranja)
    mascara_verde = cv2.inRange(hsv, intervalo_baixo_verde, intervalo_alto_verde)

    # Combina as máscaras
    # mascara = cv2.bitwise_or(mascara_laranja, mascara_verde)



    contornos,_ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contornos, -1, (0,255,0), 3)
    conts = []

    for contorno in contornos:
        (x,y), raio = cv2.minEnclosingCircle(contorno)
        centro = (int(x), int(y))
        raio = int(raio)

        if raio > 2 and y < 0.5 * frame.shape[0]:
            cv2.circle(frame, centro, raio, (0,255,0), 2)
            conts += [centro]

    list_balls += [conts]

    cv2.imshow('Detecção da Bola', mascara)
    # cv2.imshow('Detecção da Bola', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open(f'landmarks/ball_{filename}.pickle', 'wb') as f:
    pickle.dump(list(list_balls), f)

cap.release()
cv2.destroyAllWindows()
