import cv2
import numpy as np
import tensorflow as tf

from ict.read_txt import *
from cnn.machrisaa_vgg16 import Vgg16

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PROJECT_PATH='F:\Pycharm_Projects\cnn_video_representation'
import sys,os
sys.path.append(PROJECT_PATH)

RECORD_FILE = var.GIST_VIOLENCE_PATH + '/gist_violence.tfrecords'
sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
BATCH_SIZE = 32
CAPACITY = 1000
INPUT_SIZE = [-1, 224,224,3]


VALID_X_FILE = os.path.join(var.GIST_VIOLENCE_PATH,'validationX')
VALID_Y_FILE = os.path.join(var.GIST_VIOLENCE_PATH,'validationY')

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def not_zero(x):
    if x < 0:
        return 0
    else:
        return x


def make_validation_set():
    src_path = var.GIST_VIOLENCE_PATH

    n_frame = 13000
    for sample in sample_list:
        print(sample, end=' ', flush=True)

        featureX, labelY = [], []
        cap, props = im.read_video_by_path(src_path, sample + '.avi')
        bboxX = get_bbox(src_path, sample)  # bounding-box of video
        labelX = frame_label(sample)
        i = 0
        keys = list(bboxX.keys())

        while 1:
            ret, frame = cap.read()
            i += 1
            if ret is False:
                break

            if var.DEBUG:
                tools.progress_bar(i, n_frame, 10)

            if (i in keys):
                bbox = bboxX[i]
                label = labelX[i]

                one_hot = np.zeros(2, dtype=np.uint8)
                if label == 0:
                    one_hot[0] = 1
                else:
                    one_hot[1] = 1

                for k in bbox.keys():
                    x1, x2, y1, y2 = bbox[k]
                    height = y2 - y1
                    height = height * 1.1
                    height2 = int(height / 2)
                    cx = int((x1 + x2) / 2)

                    x1_ = cx - height2
                    x2_ = cx + height2

                    x1_ = not_zero(x1_)
                    x2_ = not_zero(x2_)

                    roi = cv2.resize(frame[y1:y2, x1_:x2_], (224, 224))
                    featureX.append(roi)
                    labelY.append(one_hot)

        pickle.dump(featureX,open(VALID_X_FILE,'wb'))
        pickle.dump(np.array(labelY),open(VALID_Y_FILE,'wb'))


def make_tensor_record_all():
    writer = tf.python_io.TFRecordWriter(RECORD_FILE)
    src_path = var.GIST_VIOLENCE_PATH

    n_frame = 13000

    neg_count , pos_count = 0, 0

    samples_ = [ sample_list[i] for i in [ 0,1,2, 3,4,5]]


    for sample in samples_:
        print(sample, end=' ', flush=True)

        cap, props = im.read_video_by_path(src_path, sample + '.avi')
        bboxX = get_bbox(src_path, sample)  # bounding-box of video
        labelX = frame_label(sample)
        i = 0
        keys = list(bboxX.keys())

        while 1:
            ret, frame = cap.read()
            i += 1
            if ret is False:
                break

            if var.DEBUG:
                tools.progress_bar(i, n_frame, 10)

            if (i in keys):
                bbox = bboxX[i]
                label = labelX[i]

                for k in bbox.keys():

                    if neg_count * 4 < pos_count:
                        if label ==0:
                            continue

                    x1, x2, y1, y2 = bbox[k]
                    height = y2 - y1
                    height = height * 1.1
                    height2 = int(height / 2)
                    cx = int((x1 + x2) / 2)

                    x1_ = cx - height2
                    x2_ = cx + height2

                    x1_ = not_zero(x1_)
                    x2_ = not_zero(x2_)

                    roi = cv2.resize(frame[y1:y2, x1_:x2_], (224, 224))


                    one_hot = np.zeros(2, dtype=np.uint8)
                    if label == 0:
                        one_hot[0] = 1
                        pos_count += 1
                    else:
                        one_hot[1] = 1
                        neg_count +=1


                    write_tensor(writer, roi, one_hot)

        cap.release()
    print('Abnormal count %d, Normal count %d'%(neg_count, pos_count))

def write_tensor(writer, x, y):
    img_raw = x.tostring()
    annotation_raw = y.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(img_raw),
        'annotation': _bytes_feature(annotation_raw)}))

    writer.write(example.SerializeToString())

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,

        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'annotation': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['annotation'], tf.uint8)
    image.set_shape([224 * 224 * 3])
    label.set_shape([2])

    return image, label

def inputs(record, epochs=None):
    filename_queue = tf.train.string_input_producer([record], num_epochs=epochs)

    image, label = read_and_decode(filename_queue)

    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE, num_threads=15,
        capacity=50000 + 3 * BATCH_SIZE,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=50000)
    return images, labels


def init_weights(shape, given_dev=0.01):
    return tf.Variable(tf.random_normal(shape, stddev=given_dev))


def save_weights(fc9w, fc9b):
    print('Saving tuned weights')
    data_dict = np.load(var.WEIGHT_PATH)
    keys = sorted(data_dict.keys())
    weights = []
    for k in keys:
        weights.append(data_dict[k])
    weights.append(fc9w)
    weights.append(fc9b)

    np.savez(var.TUNED_WEIGHT_PATH,
             conv1_1_W=weights[0],
             conv1_1_b=weights[1],
             conv1_2_W=weights[2],
             conv1_2_b=weights[3],
             conv2_1_W=weights[4],
             conv2_1_b=weights[5],
             conv2_2_W=weights[6],
             conv2_2_b=weights[7],
             conv3_1_W=weights[8],
             conv3_1_b=weights[9],
             conv3_2_W=weights[10],
             conv3_2_b=weights[11],
             conv3_3_W=weights[12],
             conv3_3_b=weights[13],
             conv4_1_W=weights[14],
             conv4_1_b=weights[15],
             conv4_2_W=weights[16],
             conv4_2_b=weights[17],
             conv4_3_W=weights[18],
             conv4_3_b=weights[19],
             conv5_1_W=weights[20],
             conv5_1_b=weights[21],
             conv5_2_W=weights[22],
             conv5_2_b=weights[23],
             conv5_3_W=weights[24],
             conv5_3_b=weights[25],
             fc6_W=weights[26],
             fc6_b=weights[27],
             fc7_W=weights[28],
             fc7_b=weights[29],
             fc8_W=weights[30],
             fc8_b=weights[31],
             fc9_W=weights[32],
             fc9_b=weights[33])
    print('Done ')


def train_start():
    images, labels = inputs(RECORD_FILE, 3)

    X = tf.placeholder(tf.float32, [None, 224,224, 3], name='input-X')
    Y = tf.placeholder(tf.uint8, [None, 2], name='input-Y')


    validx=  pickle.load(open(VALID_X_FILE, 'rb'))[0:64]
    validy=  pickle.load(open(VALID_Y_FILE, 'rb'))
    validy = np.array(validy,dtype=np.uint8)[0:64]

    LOG_PATH = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH,'log'))
    LOG_FILE = os.path.join(LOG_PATH,'training_log')


    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

        vgg = Vgg16(X,isTrain=True)
        logit = vgg.build()


        saver = tf.train.Saver()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost_op)
        # train_op = tf.train.AdamOptimizer(0.0001).minimize(cost_op)

        correct_prediction = tf.equal(tf.argmax(validy, 1), tf.argmax(logit, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('cost', cost_op)  # tf.scalar_summary('cost', cost_op)
        tf.summary.scalar('accuracy', acc_op)
        vgg.assigne(sess)


        try:
            step = 0
            while not coord.should_stop():
                step +=1
                _trX, _trY = sess.run([images, labels])
                trX = _trX.reshape((BATCH_SIZE, 224,224,3))
                trY = _trY.reshape((BATCH_SIZE, 2))

                count = np.count_nonzero(np.argmax(trY,axis=1))
                ratio = count/64
                # print(count, ratio)

                sess.run(train_op, feed_dict={X:trX, Y:trY})
                # loss,acc = sess.run([cost_op, acc_op], feed_dict={X:validx, Y:validy})
                m, loss = sess.run([merged, cost_op], feed_dict={X:validx, Y:validy})
                # m, acc, loss = sess.run([merged, acc_op, cost_op], feed_dict={X:validx, Y:validy})
                writer.add_summary(m, step)
                print('abnormal data ratio %0.3f, loss %0.3f'%(ratio, loss))
                # print('abnormal data ratio %0.3f, loss %0.3f, acc %0.3f'%(ratio, loss,acc))
                # print('loss %0.3f, acc %0.2f'%(loss,acc))


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
        coord.join(threads)

        fc9w, fc9b = sess.run([vgg.fc9_W, vgg.fc9_b])
        save_weights(fc9w, fc9b)
        saver.save(sess, var.PROJECT_PATH + '/model_final' + '.ckpt')


def testing():

    X = tf.placeholder(tf.float32, [None, 224,224, 3], name='input-X')
    Y = tf.placeholder(tf.uint8, [None, 2], name='input-Y')

    validx=  pickle.load(open(VALID_X_FILE, 'rb'))
    validy=  pickle.load(open(VALID_Y_FILE, 'rb'))
    validy = np.array(validy,dtype=np.uint8)

    with tf.Session() as sess:

        vgg = Vgg16(X, isTrain=False)
        logit = vgg.build()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        size =len(validx)
        STEP = 64
        step = 0

        predicted, label = [],[] # set of one-hot

        for i in range(0,size,64):
            step +=1
            tmpx = validx[i: i+STEP]
            tmpy = validy[i: i+STEP]
            pred = sess.run(logit, feed_dict={X:tmpx, Y:tmpy})

            predicted.append(pred)
            label.append(tmpy)

        acc_sperate(predicted,label)
        plotting(predicted,label)


def acc_sperate(_pred, _label):
    """
    
    :param _pred: one-hot form 
    :param _label: one-hot form 
    :return: 
    """

    pred = np.vstack(_pred)
    label = np.vstack(_label)

    pred_ = np.argmax(pred, axis=1)
    label_ = np.argmax(label, axis=1)

    abnormal_idx = np.where(label_ == 1)
    normal_idx = np.where(label_ == 0)


    abnormal_predict = pred_[abnormal_idx]
    abnormal_label = label_ [abnormal_idx]

    normal_predict = pred_[normal_idx]
    normal_label = label_[normal_idx]

    if len(abnormal_predict) == 0:
        abnormal_acc = -1
    else:
        abnormal_acc = np.mean(np.equal(abnormal_predict, abnormal_label).astype(np.uint8))

    normal_acc = np.mean(np.equal(normal_predict, normal_label).astype(np.uint8))
    mean_acc = np.mean(np.equal(pred_, label_).astype(np.uint8))
    print('abnormal acc: %0.2f  normal acc %0.2f'%(abnormal_acc, normal_acc))


def plotting(_X, _Y):


    X = np.vstack(_X)
    Y = np.vstack(_Y)
    Y = np.argmax(Y,axis=1)

    abnormal_idx = np.where(Y==1)
    abnormal_x = X[abnormal_idx]
    normal_idx = np.where(Y==0)
    normal_x= X[normal_idx]

    plt.plot(normal_x[:,0], normal_x[:,1], 'go')
    plt.plot(abnormal_x[:,0], abnormal_x[:,1], 'ro')

    red_patch = mpatches.Patch(color='red', label='The red data')
    green_patch = mpatches.Patch(color='green', label='The red data')
    plt.legend(handles=[red_patch,green_patch])
    # plt.title(self.hps)
    plt.tight_layout()
    plt.show()
    plt.clf()



def visualize():
    from time import time
    sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
    src_path = var.GIST_VIOLENCE_PATH

    total_fes, total_pos, total_neg = 0, 0, 0


    X = tf.placeholder(tf.float32, [None, 224,224, 3], name='input-X')
    Y = tf.placeholder(tf.uint8, [None, 2], name='input-Y')

    with tf.Session() as sess:

        vgg = Vgg16(X, isTrain=False)
        logit = vgg.build()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)


        predicted, label = [], []  # set of one-hot
        for sample_nxt in sample_list:  # for each video
            print(sample_nxt, end=' ', flush=True)
            path = os.path.join(var.GIST_VIOLENCE_PATH, 'predicted', 'tuned_' + sample_nxt + '.avi')


            bboxX = get_bbox(src_path, sample_nxt)  # bounding-box of video
            labelX = frame_label(sample_nxt)
            frames, prop = im.read_video_as_list_by_path(src_path, sample_nxt + '.avi')
            frames = np.array(frames)

            size = frames[0].shape
            writer = cv2.VideoWriter(path, prop['fourcc'], prop['fps'], (size[1], size[0]))

            keys = list(bboxX.keys())
            n_frame = len(frames)

            predicted = []


            for i in range(n_frame):

                tools.progress_bar(i, n_frame, 10)
                dp = frames[i].copy()
                cv2.putText(dp, 'Index: %d ' % (i), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
                start = time()

                if i in keys:
                    fbbox = bboxX[i]  # bbox of first frame


                    y = labelX[i]

                    roisx,roisy, roisk = [], [],[]
                    for k in fbbox.keys():  # id in the first frame
                        x1, x2, y1, y2 = fbbox[k]

                        resized = cv2.resize(dp[y1:y2, x1:x2], (224, 224))
                        roisx.append(resized)
                        roisy.append(y)
                        roisk.append(k)

                    pred = sess.run(logit, feed_dict={X:roisx})

                    predicted.append(np.argmax(pred,axis=1))
                    label.append(roisy)
                    pred_ = np.argmax(pred,axis=1)
                    for p ,k,l in zip(pred_,roisk,roisy):
                        bbox = fbbox[k]
                        color = var.green if p == 0 else var.red
                        msg0 = 'L:' + str(l)
                        cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[1], bbox[3]), var.black, 2)
                        txtSize, baseLine = cv2.getTextSize(msg0, var.FONT_FACE, var.FONT_SCALE, var.thickness)
                        cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[0] + txtSize[0], bbox[2] + txtSize[1]), color,-1)
                        cv2.putText(dp, msg0, (bbox[0], bbox[2] + txtSize[1]), cv2.FONT_HERSHEY_COMPLEX,
                                var.FONT_SCALE,
                                var.white, var.thickness)

                        msg1 = 'P:' + str(p)
                        txtSize, baseLine = cv2.getTextSize(msg1, var.FONT_FACE, var.FONT_SCALE, var.thickness)
                        cv2.rectangle(dp, (bbox[0], bbox[2] + 20),
                                      (bbox[0] + txtSize[0], bbox[2] + txtSize[1] + 20), color,
                                      -1)
                        cv2.putText(dp, msg1, (bbox[0], bbox[2] + txtSize[1] + 20), cv2.FONT_HERSHEY_COMPLEX,
                                    var.FONT_SCALE,
                                    var.white, var.thickness)
                end = time() - start
                fps = 30 if end == 0 else 1 / end
                if fps > 60:
                    fps = 30
                cv2.putText(dp, 'fps: %d ' % (fps), (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
                cv2.imshow('1', dp)
                cv2.waitKey(1)

                if i in keys:
                    writer.write(dp)




    #     # print(sample_nxt, ' has %d features '%self.count)
    #     label_fea = np.array(label_fea)
    #     abnormal_idx = np.where(label_fea==0)
    #     normal_idx = np.where(label_fea ==1)
    #     predict = np.array(self.predicted)
    #
    #     abnormal_predict = predict[abnormal_idx]
    #     abnormal_label = label_fea[abnormal_idx]
    #
    #     normal_predict = predict[normal_idx]
    #     normal_label = label_fea[normal_idx]
    #
    #     abnormal_acc = np.mean(np.equal(abnormal_predict,abnormal_label).astype(np.uint8))
    #     normal_acc = np.mean(np.equal(normal_predict,normal_label).astype(np.uint8))
    #
    #     # print('abnormal acc: %0.2f  normal acc %0.2f'%(abnormal_acc, normal_acc))
    #     path = os.path.join(var.GIST_VIOLENCE_PATH, 'predicted', fea_level + '_' + sample_nxt)
    #     pickle.dump(predict, open(path,'wb'))
    #     print('abnormal acc: %0.2f' %abnormal_acc, end=' ' , flush=True)
    #     print('normal acc: %0.2f' %normal_acc)
    #
    #
    # print('total  pos %s , neg %s' % (format(total_pos, ','), format(total_neg, ',')))



# make_tensor_record_all()
# make_validation_set()
# train_start()
# testing()
visualize()