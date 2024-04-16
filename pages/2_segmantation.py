import logging
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

logger = logging.getLogger(__name__)


cache_key = "yolov8-seg"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO("yolov8n-seg.pt")
    st.session_state[cache_key] = net

st.title("Realtime Segmentation YOLOv8")
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    results = net.predict(img)
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