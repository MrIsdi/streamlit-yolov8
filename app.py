import logging

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

logger = logging.getLogger(__name__)

CLASSES = [
    "Bermasker",
    "Tidak_Bermasker",
]

@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()

cache_key = "yolov8"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromONNX("best.onnx")
    st.session_state[cache_key] = net

score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    blob = cv2.dnn.blobFromImage(img, 1/255 , (640, 640), swapRB=True, mean=(0,0,0), crop= False)
    net.setInput(blob)
    outputs= net.forward(net.getUnconnectedOutLayersNames())
    out= outputs[0]

    n_detections= out.shape[1]
    height, width= img.shape[:2]
    x_scale= width/640
    y_scale= height/640
    nms_threshold= 0.5

    class_ids=[]
    score=[]
    boxes=[]

    for i in range(n_detections):
        detect=out[0][i]
        confidence= detect[4]
        if confidence >= conf_threshold:
            class_score= detect[5:]
            class_id= np.argmax(class_score)
            if (class_score[class_id]> score_threshold):
                score.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                left= int((x - w/2)* x_scale )
                top= int((y - h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h *y_scale)
                box= np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)
    for i in indices:
        color = COLORS[class_id[i]]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3] 
        cv2.rectangle(img, (left, top), (left + width, top + height), color, 3)
        label = "{}:{:.2f}".format(CLASSES[class_ids[i]], score[i])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(img, (left, top), (left + dim[0], top + dim[1] + baseline), color, 2)
        cv2.putText(
            img, 
            label, 
            (left, top + dim[1]), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)