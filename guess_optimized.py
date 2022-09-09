from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv
import sys
import cv2
from tensorflow.image import per_image_standardization as standardize

RESIZE_FINAL = 227
GENDER_LIST = ['Male','Female']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

SKIP_FRAME       = 1000
AGE_THRESHOLD    = 0.5
GENDER_THRESHOLD = 0.98

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'age',
                           'Classification type (age|gender)')


tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'dlib_opt', 'Face detection model type (yolo_tiny|cascade)')

tf.app.flags.DEFINE_string('video_file', '','Path to video file')

FLAGS = tf.app.flags.FLAGS

def one_of(fname, types):
    return any([fname.endswith('.' + ty) for ty in types])

def resolve_file(fname):
    if os.path.exists(fname): return fname
    for suffix in ('.jpg', '.png', '.JPG', '.PNG', '.jpeg'):
        cand = fname + suffix
        if os.path.exists(cand):
            return cand
    return None

def preprocess_age(image, bbox):
    # Crop and preprocess image
    crop_img = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    crop_img = cv2.resize(crop_img, (227, 227))
    tf_tensor = tf.convert_to_tensor(crop_img, dtype=tf.float32)
    output = standardize(tf_tensor)

    return output

def classify_age(age_sess, label_list, softmax_output, bbox, img, images):
    # Preprocess incoming image given bbox
    input_tf_image = preprocess_age(img, bbox)
    # Convert tf tensor to numpy
    input_np_image = input_tf_image.eval(session=age_sess)
    input_np_batch = [input_np_image]

    print("[ INFO ] Running age classification network")
    t0 = time.time()
    batch_results = age_sess.run(softmax_output, feed_dict={images:input_np_batch})
    t1 = time.time()
    t_ms = (t1 - t0) * 100
    print(f'[ INFO ] Age Classification: {t_ms} ms')

    output   = batch_results[0]
    batch_sz = batch_results.shape[0]

    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (label_list[best], output[best])
    print('[ INFO ] Guess @ 1 %s, prob = %.2f' % best_choice)
    
    out_image = img
    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)
        print('[ INFO ] Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

    if best_choice[1] > AGE_THRESHOLD:
        # Annotate image given coordinates
        result = "Age: {}, Conf: {} %".format(label_list[best], round(best_choice[1] * 100, 2))
        cv2.rectangle(out_image, (bbox[0] - 20, bbox[1] - 70), (bbox[0]+340, bbox[1] - 20), (255,255,255), -1)
        cv2.putText(out_image, str(result), (bbox[0] - 15, bbox[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    return out_image


def classify_gender(label_list, img, bbox, net):
    crop_face = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    blob=cv2.dnn.blobFromImage(crop_face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

    print("[ INFO ] Running gender classification model")
    t0 = time.time()
    net.setInput(blob)
    gender_preds = net.forward()
    gender = label_list[gender_preds[0].argmax()]
    conf = gender_preds[0][gender_preds[0].argmax()]
    t1 = time.time()
    t_ms = (t1 - t0) * 100
    print(f'[ INFO ] Gender Classification: {t_ms} ms')
    print(f'[ INFO ] Gender: {gender}')

    out_image = img
    if conf > GENDER_THRESHOLD:
        # Annotate image given coordinates
        result = "Gender: {}, Conf: {} %".format(gender, round(conf * 100, 2))
        cv2.rectangle(out_image, (bbox[0] - 20, bbox[1] - 45), (bbox[0]+340, bbox[1]-20), (255,255,255), -1)
        cv2.putText(out_image, str(result), (bbox[0] - 15, bbox[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    return out_image


def list_images(srcfile):
    with open(srcfile, 'r') as csvfile:
        delim = ',' if srcfile.endswith('.csv') else '\t'
        reader = csv.reader(csvfile, delimiter=delim)
        if srcfile.endswith('.csv') or srcfile.endswith('.tsv'):
            print('skipping header')
            _ = next(reader)
        
        return [row[0] for row in reader]

def main(argv=None):  # pylint: disable=unused-argument
    # If video file supplied, begin processing
    if FLAGS.video_file:
        # Initialize openvc video capture
        video = cv2.VideoCapture(FLAGS.video_file)

        # Initialize face detector model
        if FLAGS.face_detection_model:
            print('[ INFO ] Using face detector (%s) %s' % (FLAGS.face_detection_type, FLAGS.face_detection_model))
            face_detect = face_detection_model(FLAGS.face_detection_type, FLAGS.face_detection_model)
            print("[ INFO ] Feeding image through dlib detector for initialization")
            img = cv2.imread("man1_crop.jpg")
            img, bboxes = face_detect.run(img)
            print("[ INFO ] Complete")

        label_list = AGE_LIST if FLAGS.class_type == 'age' else GENDER_LIST
        age_nlabels = len(AGE_LIST)
        gender_nlabels = len(GENDER_LIST)

        print("[ INFO ] Num GPUS: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print('[ INFO ] Executing on %s' % FLAGS.device_id)

        # tf.config.threading.set_intra_op_parallelism_threads(5)
        # tf.config.threading.set_inter_op_parallelism_threads(5)

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as age_sess:
            with tf.device(FLAGS.device_id):
                print("[ INFO ] Selecting inception_v3 model for age classification backbone")
                age_model_fn = select_model('inception_v3') # FLAGS.model_type

                # Test fixed batch size for inputs
                age_images = tf.placeholder(tf.float32, [1, RESIZE_FINAL, RESIZE_FINAL, 3])

                requested_step = FLAGS.requested_step if FLAGS.requested_step else None

                print("[ INFO ] Initializing age logits")
                age_logits = age_model_fn(age_nlabels, age_images, 1, False)

                print("[ INFO ] Initializing tensorflow age classification model")
                #age_checkpoint_path = '/src/rude-carnie/checkpoints/age/inception/22801/'
                age_checkpoint_path = '%s' % (FLAGS.model_dir)
                age_model_checkpoint_path, global_step = get_checkpoint(age_checkpoint_path, requested_step, FLAGS.checkpoint)
                saver = tf.train.Saver()
                saver.restore(age_sess, age_model_checkpoint_path)

                age_softmax_output = tf.nn.softmax(age_logits)

                print("[ INFO ] Initializing opencv caffe gender model")
                gender_model ='gender_net.caffemodel'
                gender_proto = 'deploy_gender.prototxt'
                gender_net = cv2.dnn.readNet(gender_model,gender_proto)

                ret = True
                # Skip first X frames
                print("[ INFO ] Skipping to frame: ", SKIP_FRAME)
                for i in range(SKIP_FRAME):
                    ret, frame = video.read()

                while(ret):
                    ret, frame = video.read()
                    if not ret:
                        print('[ ERROR ] Empty video frame, exiting...')
                        sys.exit()

                    t0 = time.time()

                    print("[ INFO ] Running dlib face detector")
                    img, bboxes = face_detect.run(frame)
                    out_image = img
                    for bbox in bboxes:
                        out_image = classify_age(age_sess, AGE_LIST, age_softmax_output, bbox, out_image, age_images)
                        out_image = classify_gender(GENDER_LIST, out_image, bbox, gender_net)

                    t1 = time.time()
                    t_ms = (t1 - t0) * 100
                    print(f'[ INFO ] Total Time: {t_ms} ms')

                    img_loc = str(video.get(cv2.CAP_PROP_POS_FRAMES)) + ".jpg"
                    print(f'[ INFO ] Writing image: {img_loc}')
                    cv2.imwrite(img_loc, out_image)

                    # Extract memory info from program
                    #print("Memory Info:")
                    #tf.config.experimental.get_memory_info('CPU:0')

if __name__ == '__main__':
    tf.app.run()
