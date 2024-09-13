from FaceScore.FaceScore import FaceScore
import os 


face_score_model = FaceScore('FaceScore')
# You can load the model locally
# face_score_model = FaceScore(path_to_checkpoint,med_config = path_to_config)
img_path = 'assets/Lecun.jpg'
face_score,box,confidences = face_score_model.get_reward(img_path)
print(f'The face score of {img_path} is {face_score}, and the bounding box of the faces is {box}')