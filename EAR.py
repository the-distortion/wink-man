from scipy.spatial.distance import euclidean

def eye_aspect_ratio(eye):
  GAIN = 1	#digital amplification
  #eye has points p1, p2, ... p6, marking clockwise.
  w1 = euclidean(eye[1], eye[5]) # |p2 - p6|
  w2 = euclidean(eye[2], eye[4]) # |p3 - p4|
  
  h = euclidean(eye[0], eye[3])  # |p1 - p4|
  ear = (w1 + w2) * GAIN / (2.0 * h) #eye aspect ratio = e.a.r.
  return ear #give me the eye, and I'll give you ear XD