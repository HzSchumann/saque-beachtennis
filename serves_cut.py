import cv2
import mediapipe as mp
import os
import pickle
from collections import deque

mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(static_image_mode=False, 
                              model_complexity=2, 
                              enable_segmentation=False, 
                              smooth_landmarks=True, 
                              min_detection_confidence=0.5, 
                              min_tracking_confidence=0.5)

filename = 'renato2.mp4'
cap = cv2.VideoCapture(filename)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

seconds = 6
frames = deque(maxlen=fps*seconds)
landmarks_frames = deque(maxlen=fps*seconds)
min_visibility = 0.9
out = None

current_frame = 0
saque_count = 0
post_condition3_frames = 0
condition1_met = condition2_met = condition3_met = False

frame_skip = 2
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    current_frame += 1
    # if current_frame % frame_skip != 0:
    #     continue

    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = width, height
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    completion_percentage = (current_frame / total_frames) * 100
    os.system("clear")
    print(f"Processamento: {completion_percentage:.2f}% concluído")


    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imagem)

    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        landmarks_frames.append(results.pose_landmarks)
        frames.append(frame)

        # Check de visibilidade
        wrist_visible = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > min_visibility
        elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > min_visibility
        shoulder_visible = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > min_visibility

        # Check da condição 1 -> braço esquerdo acima do nariz
        if wrist_visible and landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.NOSE].y:
            condition1_met = True

        # Condição 2 -> Cotovelo direito acima do ombro
        if condition1_met and elbow_visible and shoulder_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition2_met = True

        # Condição 3 -> Cotovelo direito abaixo do ombro
        if condition2_met and elbow_visible and shoulder_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition3_met = True
        
        if condition3_met == True:
            post_condition3_frames += 1

        if condition1_met and condition2_met and condition3_met and out is None:
            h, w, _ = frame.shape
            saque_count += 1
            out = cv2.VideoWriter(f'serve/{filename.split(".")[0]}_{saque_count}.mp4', fourcc, fps, (w, h))

        if out is not None:
            if post_condition3_frames >= fps * 2:
                with open(f'landmarks/landmarks_{filename.split(".")[0]}_{saque_count}.pickle', 'wb') as f:
                    pickle.dump(list(landmarks_frames), f)

                while not len(frames)==0:
                    out.write(frames.popleft())
                
                out.release()
                out = None
                print(f"Saque {saque_count} salvo")
                condition1_met = condition2_met = condition3_met = False
                post_condition3_frames = 0
                    
        # Se 3 condições forem validadas, grava mais 2 segundos
        # Exporta frames e landmarks


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
