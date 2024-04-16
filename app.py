import logging
import queue
from typing import List, NamedTuple

import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class Detection(NamedTuple):
    class_id: int
    label: str
    score: str

cache_key = "yolov8"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO("yolov8n.pt")
    st.session_state[cache_key] = net

st.title("Realtime Object Detection YOLOv8")
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    results = net.predict(img)
    classId = [ int(i) for i in list(results[0].boxes.cls) ]
    conf = [ f'{float(i)*100:.2f}' for i in list(results[0].boxes.conf)]
    label = [ results[0].names[i] for i in classId]
    
    detections = [
        Detection(
            class_id=classId[i],
            label=label[i],
            score=conf[i],
        )
        for i in range(len(classId))
    ]
    
    result_queue.put(detections)
    img = results[0].plot()
    
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

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)