import streamlit as st
import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
import pdb
from scipy.spatial import distance

st.set_page_config(layout='wide')

if "idx" not in st.session_state:
    st.session_state.idx = 0
if "idx2" not in st.session_state:
    st.session_state.idx2 = 0 
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

def draw_sidebar(landmarks, landmarks2):
    expander_frame = st.sidebar.expander("Frame Controller", True)
    slider_play = expander_frame.empty()
    slider_play2 = expander_frame.empty()
    col1, col2 = expander_frame.columns(2)
    col3, col4 = st.sidebar.columns(2)

    if col1.button("-1"):
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else 0
    if col2.button("+1"):
        st.session_state.idx = st.session_state.idx + 1 if st.session_state.idx < len(landmarks) else st.session_state.idx
    if col1.button("-1 R"):
        st.session_state.idx2 = st.session_state.idx2 - 1 if st.session_state.idx2 > 0 else 0
    if col2.button("+1 R"):
        st.session_state.idx2 = st.session_state.idx2 + 1 if st.session_state.idx2 < len(landmarks2) else st.session_state.idx2

    if col1.button("Voltar"):
        st.session_state.is_playing = False
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else 0
        st.session_state.idx2 = st.session_state.idx2 - 1 if st.session_state.idx2 > 0 else 0
    
    if col2.button("Avançar"):
        st.session_state.is_playing = False
        st.session_state.idx = st.session_state.idx + 1 if st.session_state.idx < len(landmarks) else len(landmarks) - 1
        st.session_state.idx2 = st.session_state.idx2 + 1 if st.session_state.idx2 < len(landmarks2) else len(landmarks2) - 1

    if col3.button("Play"):
        st.session_state.is_playing = True
    if col4.button("Pause"):
        st.session_state.is_playing = False

    st.session_state.idx = slider_play.slider("Vídeo 1", 0, len(landmarks)-1, st.session_state.idx)
    st.session_state.idx2 = slider_play2.slider("Vídeo 2", 0, len(landmarks2)-1, st.session_state.idx2)


@st.cache_data()
def load_data(video, video2):
        land_file = f"landmarks/landmarks_{video.split('.')[0]}.pickle"
        with open(land_file, 'rb') as f:
            land_data = pickle.load(f)

        land_file = f"landmarks/landmarks_{video2.split('.')[0]}.pickle"
        with open(land_file, 'rb') as f:
            land_data2 = pickle.load(f)

        land_file = f"landmarks/ball_{video.split('.')[0]}.pickle"
        with open(land_file, 'rb') as f:
            ball = pickle.load(f)

        land_file = f"landmarks/ball_{video2.split('.')[0]}.pickle"
        with open(land_file, 'rb') as f:
            ball2 = pickle.load(f)

        return land_data, land_data2, ball, ball2

@st.cache_resource
def load_video(video, video2):
    cap = {
        1: cv2.VideoCapture(f'serve/{video}.mp4'),
        2: cv2.VideoCapture(f'serve/{video2}.mp4')
    }
    return cap


def draw_shadows(frame, landmark, idx, bp, len_):
    idx0 = max(idx - len_, 0)
    h, w, _ = frame.shape
    line_points = [(int(i.landmark[bp].x * w), int(i.landmark[bp].y * h)) for i in landmark[idx0: idx]]

    for i in range(1, len(line_points)):
        thickness = int(np.sqrt(50 / float(i + 1)) * 2.5)
        cv2.line(frame, line_points[i-1], line_points[i], (255, 255, 30), thickness)
    return frame


def draw_ball_shadow(frame, ball, idx):
    idx0 = max(idx - 20, 0)
    line_points = [i for i in ball[idx0: idx]]
    for i in range(1, len(line_points)):
        thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)

        bs0 = line_points[i-1]
        bs1 = line_points[i]
        if len(bs0) > 0 and len(bs1) > 0:
            for p0 in bs0:
                min_dist = 30
                p1_min = None
                for p1 in bs1:
                    dst = distance.euclidean(p0,p1)
                    if dst < min_dist:
                        p1_min = p1
                if p1_min is not None:
                    cv2.line(frame, p0, p1_min, (20, 200, 255), thickness)
    return frame
    pass


video_files = [i.split(".")[0].replace("ball_", "") for i in os.listdir("landmarks") if "ball" in i]
video_files.sort()

video = st.sidebar.selectbox("Selecione o vídeo 1:", video_files)
video2 = st.sidebar.selectbox("Selecione o vídeo 2:", video_files, index=len(video_files) - 1)

render_ball = st.checkbox("Render Ball?")

body_parts = {
    "right upper": 16,
    "left upper": 15,
    "right lower": 28,
    "left lower": 27,
}

bp = st.multiselect("Shadow on", body_parts.keys(), ["right upper"])

cap = load_video(video, video2)
lands_data, lands_data2, ball, ball2 = load_data(video, video2)

draw_sidebar(lands_data, lands_data2)

col1, col2, col3 = st.columns(3)
ph = col1.empty()
cont = ph.container()
ph2 = col2.empty()
cont2 = ph2.container()
ph3 = col3.empty()
cont3 = ph3.container()

#Controle Manual

if not st.session_state["is_playing"]:
    cap[1].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
    cap[2].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])
    ret, frame = cap[1].read()
    ret2, frame2 = cap[2].read()

    h, w, _ = frame.shape
    h2, w2, _ = frame2.shape

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    for y in bp:
        frame = draw_shadows(frame, lands_data, st.session_state["idx"], body_parts[y], 20)
        frame2 = draw_shadows(frame2, lands_data2, st.session_state["idx2"], body_parts[y], 20)

    if render_ball:
        frame = draw_ball_shadow(frame, ball, st.session_state["idx"])
        frame2 = draw_ball_shadow(frame2, ball2, st.session_state["idx2"])

    cont.image(frame)
    cont2.image(frame2)

while st.session_state["is_playing"]:
    st.session_state["idx"] += 1 if st.session_state["idx"] < len(lands_data) - 1 else 0
    st.session_state["idx2"] += 1 if st.session_state["idx2"] < len(lands_data2) - 1 else 0

    cap[1].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx"])
    cap[2].set(cv2.CAP_PROP_POS_FRAMES, st.session_state["idx2"])

    ret, frame = cap[1].read()
    ret2, frame2 = cap[2].read()

    h, w, _ = frame.shape
    h2, w2, _ = frame2.shape

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    for y in bp:
        frame = draw_shadows(frame, lands_data, st.session_state["idx"], body_parts[y], 20)
        frame2 = draw_shadows(frame2, lands_data2, st.session_state["idx2"], body_parts[y], 20)

    if render_ball:
        frame = draw_ball_shadow(frame, ball, st.session_state["idx"])
        frame2 = draw_ball_shadow(frame2, ball2, st.session_state["idx2"])

    with ph.container() as p:
        st.image(frame)
    with ph2.container() as p:
        st.image(frame2)