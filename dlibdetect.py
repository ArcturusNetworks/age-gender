from detect import ObjectDetector

import dlib
import cv2
import time
FACE_PAD = 50

class FaceDetectorDlib(ObjectDetector):
    def __init__(self, model_name, basename='frontal-face', tgtdir='.'):
        self.tgtdir = tgtdir
        self.basename = basename
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_name)

    def run(self, image_file):
        print(image_file)
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        images = []
        bb = []
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            bb.append((x,y,w,h))
            images.append(self.sub_image('%s/%s-%d.jpg' % (self.tgtdir, self.basename, i + 1), img, x, y, w, h))

        print('%d faces detected' % len(images))

        for (x, y, w, h) in bb:
            self.draw_rect(img, x, y, w, h)
            # Fix in case nothing found in the image
        outfile = '%s/%s.jpg' % (self.tgtdir, self.basename)
        cv2.imwrite(outfile, img)
        return images, outfile

    def sub_image(self, name, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        cv2.imwrite(name, roi_color)
        return name

    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)

# Optimized dlib detector that utilizes CNN and offloads to gpu.
# No longer requires reading of image, image is now passed as an argument.
# No longer writes image, returns bounding box data and image instead for
# further processing.
class FaceDetectorDlibOpt(ObjectDetector):
    def __init__(self, model_name, basename='frontal-face', tgtdir='.'):
        self.tgtdir = tgtdir
        self.basename = basename
        self.detector = dlib.cnn_face_detection_model_v1(model_name)

    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.time()
        faces = self.detector(gray, 1)
        t1 = time.time()
        t_ms = (t1 - t0) * 100
        print(f'[ INFO ] Dlib Face Detector: {t_ms} ms')

        bb = []
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            bb.append((x,y,w,h))

        print('[ INFO ] %d faces detected' % len(bb))

        for (x, y, w, h) in bb:
            self.draw_rect(img, x, y, w, h)
            # Fix in case nothing found in the image

        return img, bb

    def sub_image(self, name, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        cv2.imwrite(name, roi_color)
        return name

    def draw_rect(self, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)

