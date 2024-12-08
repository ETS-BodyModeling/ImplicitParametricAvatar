from deepface import DeepFace
objs = DeepFace.analyze(
  img_path = "female-1-casual_Moment.jpg",
  actions = ['gender'],
  # actions = ['age', 'gender', 'race', 'emotion'],
)
deep_gender = objs[0]["dominant_gender"]
gender = "female" if deep_gender == "Woman" else "male"
print(gender)