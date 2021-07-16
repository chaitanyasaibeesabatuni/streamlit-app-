import streamlit as st
import cv2
import numpy as np
from PIL import Image , ImageEnhance
import mediapipe as mp
import os

def hand():
  import cv2
  import mediapipe as mp

  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands

  cap = cv2.VideoCapture(0)
  with mp_hands.Hands(
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        print("your frame is not found! .")
        continue

      frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
      frame.flags.writeable = False
      results = hands.process(frame)

      frame.flags.writeable = True

      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      cv2.imshow('your Hands', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  cap.release()


face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('frecog/haarcascade_smile.xml')


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 20)

    canny_edges = cv2.Canny(img_gray_blur, 15, 70)
    ret, mask = cv2.threshold(canny_edges, 0, 255, cv2.THRESH_BINARY)

    return mask

cap = cv2.VideoCapture(0)

st.title("Image Manipulation App")
st.subheader("image and video operations by opencv")

st.write("This application performs various operations based on open cv ")


activities = ["About","Face Recognition","Cartonize","Live sketch","Hand Recognition","Eye Detector","Cannize",
                                    "Face Detector"]
choice = st.sidebar.selectbox("Select Activty", activities)


if choice == 'About':
    st.subheader("About Multi operations on Images ")
    st.markdown("Built with Streamlit")
    st.text('''    
    By:
    -Beesabathuni Chaitanya Sai''')
    st.success("Basic Operation of OpenCv")

elif choice == 'Face Recognition':

    st.subheader("Face Detection")

    image_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        # st.write(type(our_image))
        st.image(our_image)
        enhance_type = st.sidebar.radio("Enhance Type",
                                        ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # st.write(new_img)
            st.image(gray)
        elif enhance_type == 'Contrast':
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        elif enhance_type == 'Brightness':
            c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        elif enhance_type == 'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            img = cv2.cvtColor(new_img, 1)
            blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            st.image(blur_img)

        elif enhance_type == 'Original':
            st.image(our_image, width=300)
        else:
            st.image(our_image, width=300)

elif choice == 'Cartonize':
    st.subheader("Cartonize  :")
    st.write("cartonize image is : ")
    image_file = st.sidebar.file_uploader("Upload your Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.write(type(our_image))
        st.image(our_image)
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        st.image(cartoon)

elif choice == "Live sketch":
    st.subheader("Live sketch video is  :")
    st.write("press 'q' button to abort the video : ")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imwrite("livesketch.jpg", sketch(frame))
        cv2.imshow("Live Sketcher", sketch(frame))
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif choice == "Hand Recognition":
    st.subheader(" show your hand to camera :")
    st.write("press 'q' button to abort the video : ")
    hand()

elif choice == "Face Detector":
    st.subheader("your webcam video is  :")
    st.write("press 'q' button to abort the video : ")

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("video not found")
                continue

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_detection.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
            cv2.imshow('Face ', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

elif choice == 'Cannize':
    st.subheader("Cannize  :")
    st.write("cannize image is : ")
    image_file = st.sidebar.file_uploader("Upload your Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.write(type(our_image))
        st.image(our_image)
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        canny = cv2.Canny(img, 100, 150)
        st.image(canny)

elif choice == 'Eye Detector':
    eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')
    st.subheader("Eye Detector  :")
    st.write("Eye on image is : ")
    image_file = st.sidebar.file_uploader("Upload your Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.write(type(our_image))
        st.image(our_image)
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img, 1)
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    st.image(img)

