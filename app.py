import logging
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

logger = logging.getLogger(__name__)

model = YOLO("yolov8n.pt")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    result = model(image)
    
    image = result[0].plot()

    return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Realtime Object Detection Using YOLOv8")

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)