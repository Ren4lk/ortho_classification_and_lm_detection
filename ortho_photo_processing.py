import cv2
import os
import mediapipe as mp
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import numpy as np
from math import *
import imutils

from sixdrepnet import SixDRepNet
import fl_model
import oc_model

from utilities import *

import matplotlib.pyplot as plt


def imageToTensor(img: np.ndarray,
                  size: tuple[int, int]) -> torch.Tensor:
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = TF.resize(Image.fromarray(temp), size=size)
    temp = TF.to_tensor(temp)
    temp = TF.normalize(temp, [0.5], [0.5])
    return temp


def predictClass(img_path: str) -> str:
    classification_model = oc_model.Network()
    classification_model.load_state_dict(
        torch.load(ORTHO_CLASSIFICATION_WEIGHTS_PATH))
    classification_model.eval()
    image = cv2.imread(img_path)
    image = imageToTensor(img=image, size=(300, 300))

    with torch.no_grad():
        target = classification_model(image.unsqueeze(0))
    target = torch.argmax(torch.softmax(target, dim=1), dim=1)
    return IDX_TO_CLASS[target.item()]


def getProfileLandmarks(img: np.ndarray,
                        angle: float,
                        x: np.int64,
                        y: np.int64,
                        w: np.int64,
                        h: np.int64) -> np.ndarray[float]:
    landmarks_profile_model = fl_model.Network(num_classes=78)
    landmarks_profile_model.load_state_dict(
        torch.load(PROFILE_LANDMARKS_DETECTION_WEIGHTS_PATH))
    landmarks_profile_model.eval()

    img = img[y:y+h, x:x+w]
    if angle > 50:
        img = cv2.flip(img, 1)
    img_tensor = imageToTensor(img=img, size=(300, 300))

    with torch.no_grad():
        landmarks = landmarks_profile_model(img_tensor.unsqueeze(0))

    landmarks = (landmarks.view(39, 2).detach().numpy() + 0.5) * \
        np.array([[w, h]])
    if angle > 50:
        landmarks[:, 0] = w - landmarks[:, 0]  # mirroring landmarks
    landmarks += np.array([[x, y]])

    return landmarks


def getFullFaceLandmarks(img: np.ndarray,
                         w: int,
                         h: int) -> np.ndarray[float]:
    mp_face_mesh = mp.solutions.face_mesh
    landmarks = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for faces in results.multi_face_landmarks:
                for landmark in faces.landmark:
                    landmarks.append([landmark.x * w, landmark.y * h])
                landmarks = np.array(landmarks)
        else:
            raise Exception('No landmarks detected')
    return landmarks


def predictLandmarksAndAngle(img_path: str) -> tuple[np.ndarray[float],
                                                     np.float32]:
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0,
                                         min_detection_confidence=0.6) as face_detection:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not results.detections:
        raise Exception('No face detected')

    img_h, img_w = image.shape[:2]
    face_rectangle = [results.detections[0].location_data.relative_bounding_box.xmin,
                      results.detections[0].location_data.relative_bounding_box.ymin,
                      results.detections[0].location_data.relative_bounding_box.width,
                      results.detections[0].location_data.relative_bounding_box.height] * \
        np.array([img_w, img_h, img_w, img_h])
    xmin, ymin, width, height = face_rectangle.astype(np.int64)

    head_pose_model = SixDRepNet()
    _, yaw, _ = head_pose_model.predict(image[ymin:ymin+height,
                                              xmin:xmin+width])

    landmarks = []
    if yaw > 50 or yaw < -50:
        landmarks = getProfileLandmarks(img=image,
                                        angle=yaw,
                                        x=xmin,
                                        y=ymin,
                                        w=width,
                                        h=height)
    else:
        landmarks = getFullFaceLandmarks(img=image, w=img_w, h=img_h)

    return landmarks, yaw


def getFaceBox(width: int,
               height: int,
               landmarks: np.ndarray,
               angle: np.float32) -> tuple[int, int, int, int, float]:
    if angle > -50 and angle < 50:  # middle
        left_eye_middle = landmarks[468]
        right_eye_middle = landmarks[473]
        rotation_angle = np.arctan2((right_eye_middle[1] - left_eye_middle[1]),
                                    (right_eye_middle[0] - left_eye_middle[0]))
        transformation_matrix = np.array([
            [+cos(rotation_angle), -sin(rotation_angle)],
            [+sin(rotation_angle), +cos(rotation_angle)]
        ])
        landmarks = np.matmul(landmarks, transformation_matrix)
    else:
        rotation_angle = 0.0

    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bot = np.max(landmarks[:, 1])

    width_of_face = right - left
    heigh_of_face = bot - top

    if angle < -50:  # on the right
        left -= 0.8 * width_of_face
        right += 0.1 * width_of_face
        top -= 0.95 * heigh_of_face
        bot += 0.3 * heigh_of_face
    elif angle > 50:  # on the left
        left -= 0.1 * width_of_face
        right += 0.8 * width_of_face
        top -= 0.95 * heigh_of_face
        bot += 0.3 * heigh_of_face
    else:  # middle
        left -= 0.2 * width_of_face
        right += 0.2 * width_of_face
        top -= 0.4 * heigh_of_face
        bot += 0.07 * heigh_of_face

    if left < 0:
        left = 0
    if right > width:
        right = width
    if top < 0:
        top = 0
    if bot > height:
        bot = height

    return int(left), int(right), int(top), int(bot), rotation_angle


def test():
    dir = 'photo dir'
    images_paths = []
    for _, _, files in os.walk(dir):
        for filename in files:
            images_paths.append(os.path.join(dir, filename))

    for path in images_paths:
        try:
            img = cv2.imread(path)
            img_h, img_w = img.shape[:2]

            target = predictClass(path)
            print(f'TYPE: {target} || PHOTO: {path}')

            if target == 'portrait':
                landmarks, face_angle = predictLandmarksAndAngle(path)
                left, right, top, bot, rotation_angle = getFaceBox(width=img_w,
                                                                   height=img_h,
                                                                   landmarks=landmarks,
                                                                   angle=face_angle)
                img = imutils.rotate(np.array(img),
                                     np.rad2deg(rotation_angle),
                                     (0, 0))
                img = img[top:bot, left:right]

            plt.figure(figsize=(13, 13))
            # plt.scatter(landmarks[:, 0], landmarks[:, 1], c='black', s=10)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(e)
        finally:
            plt.show()


if __name__ == '__main__':
    test()
