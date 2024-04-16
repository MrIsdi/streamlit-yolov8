import logging
import streamlit as st

logger = logging.getLogger(__name__)

st.title("Portfolio Machine Learning")

col1, col2 = st.columns(2)
with col1:
    st.image("./assets/user.png", caption="Muhammad Ridho Isdi", width=300)
    
with col2:
    st.header("About me")
    st.text("""
            Machine learning development
            has been my passion from 2022 until now. 
            From learning Supervised Learning, 
            Unsupervised Learning, and Deep Learning 
            such as Object Detection from images/
            videos and audio classification.
            """)