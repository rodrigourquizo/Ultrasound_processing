import cv2
import numpy as np
import matplotlib.pyplot as plt

videofile = 'Subject_03.mp4'

video = cv2.VideoCapture(videofile)
numbFram = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#Mejora de contraste
GrayFram = np.zeros((numbFram,), dtype=object)
MPI = np.zeros((numbFram,))

for k in range(numbFram):
    ret, Frame = video.read()
    GrayFram[k] = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    MPI[k] = np.mean(GrayFram[k])

indmax = np.argmax(MPI) + 1

video.set(cv2.CAP_PROP_POS_FRAMES, indmax - 1)
_, maxFrame = video.read()
maxFrame = cv2.cvtColor(maxFrame, cv2.COLOR_BGR2GRAY)

edgeThreshold = 1
amount = 0.2
I = cv2.createCLAHE(clipLimit=amount, tileGridSize=(8, 8)).apply(maxFrame)
cv2.imwrite('mejorada.png', I)

video.release()

#Difuminar fondo
video = cv2.VideoCapture(videofile)
height, width = I.shape
accumulator = np.zeros((height, width, 3), dtype=np.float32)
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    accumulator += frame.astype(np.float32)
    frame_count += 1

average_frame = accumulator / frame_count
video.release()
video = cv2.VideoCapture(videofile)
std_dev_accumulator = np.zeros((height, width, 3), dtype=np.float32)

while True:
    ret, frame = video.read()
    if not ret:
        break
    std_dev_accumulator += (frame.astype(np.float32) - average_frame) ** 2

std_dev_frame = np.sqrt(std_dev_accumulator / frame_count)
threshold = 0.9
mask = std_dev_frame > threshold

I = cv2.imread("mejorada.png")
result_frame = np.where(mask, I.astype(np.uint8), 0)
cv2.imwrite('sin_fondo.png', result_frame)
video.release()


#Contorno y transformacion rectangular
video = cv2.VideoCapture(videofile)
ret, first_frame = video.read()
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

img = cv2.imread('sin_fondo.png')
blue, green, red = cv2.split(img)
contours, _ = cv2.findContours(image=green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
contours = list(contours)

index = 0
area_max = 160
for i, cnt in enumerate(contours): 
    area = cv2.contourArea(cnt)
    if area >= area_max:
        index = i
        area_max = area
contour = contours[index] #contorno de mayor área


leftmost  = tuple(contour[contour[:, :, 0].argmin()][0])
rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
topmost   = tuple(contour[contour[:, :, 1].argmin()][0])
bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

for p in (leftmost, rightmost, topmost, bottommost,
          (leftmost[0], rightmost[1]),
          (rightmost[0], leftmost[1]),
          (topmost[1] + 520, topmost[1])):
    cv2.circle(img, p, 5, (0, 0, 255), 2)

linea1 = [leftmost, topmost]
linea2 = [(topmost[1] + 520, topmost[1]), (rightmost[0], leftmost[1])]
(x1, y1), (x2, y2) = linea1
(x3, y3), (x4, y4) = linea2
m1, b1 = np.polyfit([x1, x2], [y1, y2], 1)
m2, b2 = np.polyfit([x3, x4], [y3, y4], 1)
intersection_x = (b2 - b1) / (m1 - m2)
intersection_y = m1 * intersection_x + b1
centro = (int(intersection_x), int(intersection_y))
cv2.circle(img, centro, 5, (0, 0, 255), -1)

point = contour[600][0]
cv2.circle(img, tuple(point), 4, (255, 255, 255), -1)
radio = int(np.hypot(point[0] - centro[0], point[1] - centro[1]))
radio_peque = int(np.hypot(topmost[0] - centro[0], topmost[1] - centro[1]))

def generate_ellipse_points(center, axes, angle, start_angle, end_angle, n):
    h, k = center; a, b = axes
    t = np.linspace(np.radians(start_angle), np.radians(end_angle), n)
    x = h + a*np.cos(t)*np.cos(angle) - b*np.sin(t)*np.sin(angle)
    y = k + a*np.cos(t)*np.sin(angle) + b*np.sin(t)*np.cos(angle)
    return np.column_stack((x, y)).astype(int)

ellipse_points      = generate_ellipse_points(centro, (radio, radio), 0, 50, 130, 1000)
ellipse_upper_points = generate_ellipse_points(centro, (radio_peque, radio_peque), 0, 50, 130, 1000)
coords = [np.column_stack((np.linspace(p2[0], p1[0], 1000, dtype=int).clip(0, img.shape[1]-1),
                           np.linspace(p2[1], p1[1], 1000, dtype=int).clip(0, img.shape[0]-1)))
          for p1, p2 in zip(ellipse_points, ellipse_upper_points)]

pixel_values = [gray[c[:,1], c[:,0]] for c in coords]
max_length = max(len(v) for v in pixel_values)
pixel_matrix = np.zeros((max_length, len(pixel_values)), dtype=np.uint8)
for i, v in enumerate(pixel_values):
    pixel_matrix[:len(v), i] = v

h, w = pixel_matrix.shape
cap = cv2.VideoCapture(videofile)
video_out = cv2.VideoWriter('results.mp4', -1, 1, (max_length + h, len(pixel_values)))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixel_values = [gray[c[:,1], c[:,0]] for c in coords]
    max_length = max(len(v) for v in pixel_values)
    pixel_matrix = np.zeros((max_length, len(pixel_values)), dtype=np.uint8)
    for i, v in enumerate(pixel_values):
        pixel_matrix[:len(v), i] = v
    resized_gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    new = np.concatenate((resized_gray, pixel_matrix), axis=1)
    video_out.write(new)

cap.release()
video_out.release()

cv2.drawContours(img, [contour], -1, (0, 255, 0), 2) 
cv2.imwrite('contorno_pintado.png', img)
