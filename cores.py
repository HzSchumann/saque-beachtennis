import cv2
import numpy as np

# Variável de controle para pausar e continuar o vídeo
paused = False

# Função para capturar o clique do mouse
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_hsv = hsv[y, x]
        print(f"Cor HSV selecionada: {color_hsv}")

filename = "renato1_1"
# cap = cv2.VideoCapture(f'serve/{filename}.mp4')
cap = cv2.VideoCapture(f'serve/{filename}.mp4')

# Cria uma janela e define a função de callback para o clique do mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', pick_color)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Exibe o vídeo
        cv2.imshow('Video', frame)

    # Espera por uma tecla pressionada por 30ms
    key = cv2.waitKey(30) & 0xFF
    
    # Pressiona 'q' para sair do loop
    if key == ord('q'):
        break
    # Pressiona 'p' para pausar/resumir o vídeo
    elif key == ord('p'):
        paused = not paused

cap.release()
cv2.destroyAllWindows()


