import mtcnn
import matplotlib.pyplot as plt
import imutils
import math
import numpy as np

filename = "Photos/test2.jpg"
image = plt.imread(filename)

detector = mtcnn.MTCNN()
faces = detector.detect_faces(image)

def alignment_process(image, faces):
    x1, y1 = faces[0]['keypoints']['right_eye']
    x2, y2 = faces[0]['keypoints']['left_eye']
    print(x1, y1)
    print(x2, y2)

    a = abs(y1 - y2)
    b = abs(x2 - x1)
    c = math.sqrt(a*a + b*b)

    cos_alpha = (b*b + c*c - a*a) / (2*b*c)

    alpha = np.arccos(cos_alpha)
    alpha = (alpha * 180) / math.pi

    aligned_img = imutils.rotate(image, angle=alpha)
    plt.imshow(aligned_img)
    plt.show()

def draw_facebox(image, faces):
    plt.imshow(image)
    ax = plt.gca()

    for face in faces:
        x, y, width, height = face['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)

    plt.show()

def draw_faces(image, faces):
    for i in range(len(faces)):
        x1, y1, width, height = faces[i]['box']
        x2, y2 = x1 + width, y1 + height

        plt.subplot(1, len(faces), i+1)
        plt.axis('off')

        plt.imshow(image[y1:y2, x1:x2])

    plt.show()

draw_facebox(image, faces)
draw_faces(image, faces)
#alignment_process(image, faces)