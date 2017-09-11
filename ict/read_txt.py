import pickle

import cv2
import numpy as np

import tiah.ImageHandler as im
import tiah.FileManager as fm
import tiah.tools as tools
import var

sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']


def img_to_video():
    for sample_nxt in sample_list:
        tmp_path = var.GIST_VIOLENCE_PATH + '/' + sample_nxt
        frames = im.read_images_by_path(tmp_path)
        _, prop = im.read_video_by_path(var.GIST_VIOLENCE_PATH, 'cam1.avi')

        size = frames[0].shape
        path = var.GIST_VIOLENCE_PATH + '/' + sample_nxt + '.avi'
        writer = cv2.VideoWriter(path, prop['fourcc'], prop['fps'], (size[1], size[0]))
        print(sample_nxt + ' len: %d' % prop['length'])

        for f in frames:
            writer.write(f)
        writer.release()


def read_txt(src_path, cam_nxt):
    f = open(src_path + '/' + cam_nxt + '.txt', 'r')

    prvs_idx = -1
    gt = []

    while True:
        line = f.readline().split('\n')[0]
        if not line: break

        line = tools.int2round(line.split(' '))
        curr_idx = line[1]  # frame dix
        id, x, y, w, h = line[3], line[4], line[5], line[6], line[7]
        bbox = (x, x + w, y, y + h)

        if prvs_idx == curr_idx:
            frame = gt[-1]
            frame[id] = bbox
        else:
            frame = {id: bbox}
            gt.append(frame)
            prvs_idx = curr_idx

    f.close()
    return gt


def get_bbox(src_path, cam_nxt):
    if fm.isExist(src_path, 'bbox_'+cam_nxt,pr=1):
    # if False:
        return pickle.load(open(src_path + '/bbox_' + cam_nxt, 'rb'))
    else:
        gt = read_txt2(src_path, cam_nxt)
        pickle.dump(gt, open(src_path + '/bbox_' + cam_nxt, 'wb'))
        return gt


def read_txt2(src_path, cam_nxt):
    f = open(src_path + '/' + cam_nxt + '.txt', 'r')

    gt = {}
    line_count, frame_count, child_count = 0, 0, 0
    while True:
        line = f.readline().split('\n')[0]
        if not line: break
        line_count += 1
        line = tools.int2round(line.split(' '))
        # line (cam, frame, object, _, x, y, w, h)
        fid, oid, x, y, w, h = line[1], line[3], line[4], line[5], line[6], line[7]
        bbox = (x, x + w, y, y + h)

        if w == 0 and h == 0:
            continue

        check_dup(gt, fid, oid)
        if fid in gt.keys():
            frame = gt[fid]
            frame[oid] = bbox
            child_count += 1
        else:
            frame_count += 1
            frame = {oid: bbox}
            gt[fid] = frame

    f.close()

    if var.DEBUG:
        print('Read text %d lines' % (line_count))
        print('unique frame count ', frame_count, ' child count ', child_count,' = ' , frame_count+child_count)

    return gt


def check_dup(gt, fid, oid):
    if fid in gt.keys():
        bb = gt[fid]
        if oid in bb.keys():
            print(fid, oid, ' duplicated!!!!!!!')
            quit()


def frame_label(cam_nxt):
    """
    -1 removed
    1  not walking
    0 walking
    
    
    :param cam_nxt: 
    :return: 
    """

    a = np.zeros(12530, dtype=np.uint8)
    if cam_nxt == 'cam1':
        a[np.arange(10415, 10520)] = -1
        a[np.arange(10800, 11160)] = 1

    elif cam_nxt == 'cam2':
        a[np.arange(9350, 9700)] = 1

    elif cam_nxt == 'cam3':
        a[np.arange(3293, 3430)] = -1
        a[np.arange(11527, 11950)] = -1
        a[np.arange(11978, 12520)] = 1

    elif cam_nxt == 'cam4':
        a[np.arange(9428, 9670)] = 1
        a[np.arange(9974, 10120)] = 1
        a[np.arange(11636, 11834)] = 1
        a[np.arange(11944, 12520)] = -1
    elif cam_nxt == 'cam6':
        a[np.arange(893, 1152)] = -1
        a[np.arange(5918, 6076)] = 1
        a[np.arange(7795, 9110)] = -1
        a[np.arange(10905, 11075)] = 1

    return a


def label_video(src_path):
    for sample_nxt in sample_list:
        gt = get_bbox(src_path, sample_nxt)
        labelX = frame_label(sample_nxt)
        frames, prop = im.read_video_as_list_by_path(src_path, sample_nxt + '.avi')

        n_frame, n_gt = len(frames), len(gt)
        # assert (n_frame == n_gt), 'frame and gt should have same length %d != %d'%(n_frame,n_gt)
        print(sample_nxt, n_frame)
        size = frames[0].shape
        path = src_path + '/labeled_' + sample_nxt + '.avi'
        writer = cv2.VideoWriter(path, prop['fourcc'], prop['fps'], (size[1], size[0]))

        white = (255, 255, 255)
        red = (0, 0, 255)
        green = (0, 255, 0)

        FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
        FONT_SCALE = 0.8
        thickness = 2
        for idx in range(len(frames)):
            frame = frames[idx]
            label = labelX[idx]

            tools.progress_bar(idx, n_frame, 10)
            cv2.putText(frame, 'Index: %d ' % (idx), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), thickness=2)

            if idx in gt.keys():
                frame_gt = gt[idx]
                for key in frame_gt.keys():
                    bbox = frame_gt[key]

                    color = green if label == 0 else red
                    msg = 'Walk' if label == 0 else '~Walk'

                    txtSize, baseLine = cv2.getTextSize(msg, FONT_FACE, FONT_SCALE, thickness)

                    cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), color, 2)
                    cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[0] + txtSize[0], bbox[2] + txtSize[1]), color, -1)
                    cv2.putText(frame, msg, (bbox[0], bbox[2] + txtSize[1]), cv2.FONT_HERSHEY_COMPLEX, FONT_SCALE,
                                white, thickness)

            # cv2.imshow('1',frame)
            # cv2.waitKey(10)
            writer.write(frame)

        writer.release()
