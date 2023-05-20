import os

ORTHO_CLASSIFICATION_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'ortho_classification_weights.pth')

ORTHO_CLASSES = ['jaw-lower',
                 'jaw-upper',
                 'mouth-sagittal_fissure',
                 'mouth-vestibule-front-closed',
                 'mouth-vestibule-front-half_open',
                 'mouth-vestibule-half_profile-closed-left',
                 'mouth-vestibule-half_profile-closed-right',
                 'mouth-vestibule-profile-closed-left',
                 'mouth-vestibule-profile-closed-right',
                 'portrait']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(ORTHO_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(ORTHO_CLASSES)}

PROFILE_LANDMARKS_DETECTION_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'resnet50_face_39_landmarks_weights_epoch_462.pth')
