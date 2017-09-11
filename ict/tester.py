import os
import pickle
from collections import namedtuple
from time import time

import tensorflow as tf

from sklearn.preprocessing import normalize

from encoding.VLAD import VLAD
from ict.read_txt import *

import tiah.FileManager as fm

HYPER_PARAMS = namedtuple('HYPER_PARAMS',
                          'spatial_size,'
                          'temporal_size,'
                          'tow,'
                          'dtv')


class runner():
    def __init__(self,hps, vgg, sess):

        self.sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']

        self.frame_container = []
        self.bbox_container = {}

        self.sess =sess
        self.vgg =vgg
        self.nn_frame = 5
        self.vlader = VLAD(hps.vlad_k)


        mpath = self.make_model_path(hps)
        print('Loading model: ' , mpath)

        self.svc_model = pickle.load(open(mpath, 'rb'))
        self.hps = hps

        vgg_level = 'vgg' + str(hps.cnn_ver)
        clip_level = 'clip'
        pca_level =  'pca' + str(hps.pca_k)
        vlad_level = pca_level + '_vlad' + str(hps.vlad_k)

        vgg_bpath = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH, vgg_level))
        self.clip_bpath = os.path.join(vgg_bpath, clip_level)
        self.pca_bpath = fm.mkdir(os.path.join(vgg_bpath,pca_level))

        self.vlad_bpath = fm.mkdir(os.path.join(vgg_bpath, vlad_level))
        self.vlad_maker = VLAD(self.hps)



    def make_model_path(self, hps):
        model_level = 'svc_' + hps.fl + '_v' + str(hps.cnn_ver) \
                      + '_c' + str(hps.clip_n) + '_v' + str(hps.vlad_k)
        tmp = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH, 'models'))
        return os.path.join(tmp, model_level)


    def running(self, y, j):
        """
        
        :param y: frame label  
        :param j: current frame index 
        :return: 
        """

        start = time()
        clip = self.hps.clip_n - 1
        i = j - clip

        keys = self.bbox_container.keys()
        dp = self.frame_container[-1].copy()
        cv2.putText(dp, 'Index: %d ' % (j), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
        tmp_path = 'f:/DATASET/13. GIST/1.violence/wow'

        if (i in keys) and ( j in keys):

            gt = self.bbox_container[j]
            fc6x = self.use_cnn(i, j)



            for k in gt.keys(): # draw all bbox in the frame.
                bbox = gt[k]
                cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[1], bbox[3]), var.black, 2)


                if k in fc6x.keys(): # if bbox has detection result.
                    self.count +=1
                    fc6 = fc6x[k]
                    vlad = self.vlader.get_vlad_descriptor(fc6)

                    pred = self.svc_model.predict(vlad.reshape(1, -1))[0]
                    self.predicted.append(pred)
                    # print('prediction: ' , pred)

                    color = var.green if pred == 0 else var.red
                    msg0 = 'L:' + str(y)

                    txtSize, baseLine = cv2.getTextSize(msg0, var.FONT_FACE, var.FONT_SCALE, var.thickness)
                    cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[0] + txtSize[0], bbox[2] + txtSize[1]), color, -1)
                    cv2.putText(dp, msg0, (bbox[0], bbox[2] + txtSize[1]), cv2.FONT_HERSHEY_COMPLEX, var.FONT_SCALE,
                                var.white, var.thickness)

                    msg1 = 'P:' + str(pred)
                    txtSize, baseLine = cv2.getTextSize(msg1, var.FONT_FACE, var.FONT_SCALE, var.thickness)
                    cv2.rectangle(dp, (bbox[0], bbox[2] + 20), (bbox[0] + txtSize[0], bbox[2] + txtSize[1] + 20), color,
                                  -1)
                    cv2.putText(dp, msg1, (bbox[0], bbox[2] + txtSize[1] + 20), cv2.FONT_HERSHEY_COMPLEX,
                                var.FONT_SCALE,
                                var.white, var.thickness)

        end = time() - start
        fps = 30 if end == 0 else 1 / end
        if fps > 60:
            fps = 30

        cv2.putText(dp, 'fps: %d ' % (fps), (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)

        return dp

    def use_cnn(self,i,j):


        prvs_bbox = self.bbox_container[i]  # bbox of first frame
        next_bbox = self.bbox_container[j]  # bbox of last frame

        fc6X = {}
        for k in next_bbox.keys():  # id in the first frame
            if k in prvs_bbox.keys():  # is still presence in the last frame


                x1, x2, y1, y2 = prvs_bbox[k]
                s1, s2, t1, t2 = next_bbox[k]

                # ROI selection
                minx, maxx = min(x1, s1, x2, s2), max(x1, s1, x2, s2),
                miny, maxy = min(y1, t1, y2, t2), max(y1, t1, y2, t2)
                print('running extraing between', i, j, k)
                print(miny,maxy, minx,maxx)
                quit()
                rois = []
                for f in self.frame_container[i: j+1]:
                    resized = cv2.resize(f[miny:maxy, minx:maxx], (224, 224))
                    rois.append(resized)

                print('use_ cnn roi size ', len(rois), ' ' , rois[0].shape)
                f6 = self.sess.run(self.vgg.fc1, feed_dict={self.vgg.imgs: rois})

                if self.hps.fl == 'nf6':
                    print('hps fl ' , self.hps.fl, ' normalize')
                    f6 = normalize(f6, axis=0)  # default is l2

                fc6X[k] = f6

        return fc6X


    def aaaaaaaaaaaaaaaa(self):
        for sample_nxt in sample_list[3:4]:  # for each video
            print('Visualizing ', sample_nxt)
            bboxX = get_bbox(var.GIST_VIOLENCE_PATH, sample_nxt)  # bounding-box of video

            frames, prop = im.read_video_as_list_by_path(var.GIST_VIOLENCE_PATH, sample_nxt + '.avi')
            path = os.path.join(var.GIST_VIOLENCE_PATH, 'predicted', sample_nxt + '.avi')


            fpath = os.path.join(var.GIST_VIOLENCE_PATH,'vgg16','clip30_vlad9','nf6', 'cam4')
            feax = pickle.load(open(fpath+'/fea_nameX','rb'))

            model_level = 'svc_nf6_' + 'v' + str(self.hps.cnn_ver) + '_c' + str(self.hps.clip_n) + '_v' + str(
                self.hps.vlad_k)
            tmp = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH, 'models'))
            mpath = os.path.join(tmp, model_level)
            model = pickle.load(open(mpath,'rb'))

            for i in range(0,1000):

                name = feax[i]
                _,fid,oid = name.split('.')[0].split('_')

                fid = int(fid)
                oid = int(oid)
                print(name, ' >> ' , fid, oid)
                frame = frames[fid]
                bbox_ = bboxX[fid]
                bbox= bbox_[oid]

                x = np.load(fpath+'/nf6_'+name)
                pred = model.predict(x.reshape(1, -1))[0]
                # print('prediction: ' , pred)

                color = var.green if pred == 0 else var.red
                msg = 'Walk' if pred == 0 else '~Walk'


                cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), var.black, 2)
                # cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[0] + txtSize[0], bbox[2] + txtSize[1]), color, -1)

                # msg0 = 'L:' + str(y)
                # txtSize, baseLine = cv2.getTextSize(msg0, var.FONT_FACE, var.FONT_SCALE, var.thickness)
                # cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[0] + txtSize[0], bbox[2] + txtSize[1]), color, -1)
                # cv2.putText(dp, msg0, (bbox[0], bbox[2] + txtSize[1]), cv2.FONT_HERSHEY_COMPLEX, var.FONT_SCALE,
                #             var.white, var.thickness)

                msg1 = 'P:' + str(pred)
                txtSize, baseLine = cv2.getTextSize(msg1, var.FONT_FACE, var.FONT_SCALE, var.thickness)
                cv2.rectangle(frame, (bbox[0], bbox[2] + 20), (bbox[0] + txtSize[0], bbox[2] + txtSize[1] + 20), color, -1)
                cv2.putText(frame, msg1, (bbox[0], bbox[2] + txtSize[1] + 20), cv2.FONT_HERSHEY_COMPLEX, var.FONT_SCALE,
                            var.white, var.thickness)


            tmp_path = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH, 'predicted','tmp'))
            im.write_video(frames[0:1000],prop['fps'],tmp_path,'bbbb', prop)



            # tmp_path = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH, 'predicted','tmp'))
            # print(path)
            # size = frames[0].shape
            # writer = cv2.VideoWriter(path, prop['fourcc'], prop['fps'], (size[1], size[0]))


    #
    # def ssibal(self):
    #
    #     hps= self.hps
    #     clip = self.hps.clip_n - 1
    #
    #
    #
    #     sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
    #     src_path = var.GIST_VIOLENCE_PATH
    #
    #     vlad_maker = VLAD(self.hps)
    #
    #     total_fes, total_pos, total_neg = 0, 0, 0
    #
    #
    #     for sample_nxt in sample_list[3:4]:  # for each video
    #         print(sample_nxt, end=' ', flush=True)
    #         path = os.path.join(var.GIST_VIOLENCE_PATH, 'predicted', fea_level + '_' + sample_nxt + '.avi')
    #
    #         out_path = fm.mkdir(os.path.join(clip_bpath, sample_nxt))
    #         label_fea = pickle.load(open(out_path + '/fea_nameY', 'rb'))
    #
    #         bboxX = get_bbox(src_path, sample_nxt)  # bounding-box of video
    #         labelX = frame_label(sample_nxt)
    #         frames, prop = im.read_video_as_list_by_path(src_path, sample_nxt + '.avi')
    #         frames = np.array(frames)
    #
    #         size = frames[0].shape
    #         writer = cv2.VideoWriter(path, prop['fourcc'], prop['fps'], (size[1], size[0]))
    #
    #         clip_cam_path = fm.mkdir(os.path.join(self.clip_bpath, sample_nxt+'ab'))
    #         vlad_cam_path = fm.mkdir(os.path.join(self.vlad_bpath, sample_nxt+'ab'))
    #         feaX, feaY = [], []
    #
    #         keys = list(bboxX.keys())
    #         n_frame = len(frames)
    #         nn_frame = len(str(n_frame))
    #         self.count = 0
    #         self.predicted = []
    #
    #         for i in range(n_frame):
    #
    #
    #             # tools.progress_bar(i, n_frame, 10)
    #             j = i + clip
    #             if (i in keys) and (j in keys):  # if roi has 10 frames
    #                 prvs_bbox = bboxX[i]  # bbox of first frame
    #                 next_bbox = bboxX[j]  # bbox of last frame
    #
    #                 dp = frames[j].copy()
    #                 cv2.putText(dp, 'Index: %d ' % (j), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
    #
    #                 y = labelX[j]
    #                 start = time()
    #
    #                 for k in prvs_bbox.keys():  # id in the first frame
    #                     if k in next_bbox.keys():  # is still presence in the last frame
    #                         bbox = next_bbox[k]
    #                         self.count += 1
    #                         x1, x2, y1, y2 = prvs_bbox[k]
    #                         s1, s2, t1, t2 = next_bbox[k]
    #
    #                         # ROI selection
    #                         minx, maxx = min(x1, s1, x2, s2), max(x1, s1, x2, s2),
    #                         miny, maxy = min(y1, t1, y2, t2), max(y1, t1, y2, t2)
    #
    #                         rois = []
    #                         for f in frames[i: j + 1]:
    #                             resized = cv2.resize(f[miny:maxy, minx:maxx], (224, 224))
    #                             rois.append(resized)
    #
    #                         f6 = self.sess.run(self.vgg.fc1, feed_dict={self.vgg.imgs: rois})
    #                         if hps.fl =='nf6':
    #                             f6 = normalize(f6, axis=0)  # default is l2
    #
    #                         vlad = vlad_maker.get_vlad_descriptor(f6)
    #
    #
    #                         # fname = str(i).zfill(nn_frame) + '_' + str(k).zfill(2) + '.npy'
    #                         # np.save(os.path.join(clip_cam_path, 'f6_' + fname), f6)
    #                         # np.save(os.path.join(clip_cam_path, 'nf6_' + fname), nf6)
    #                         # np.save(os.path.join(vlad_cam_path, 'vlad_f6_' + fname), vlad_f6)
    #                         # np.save(os.path.join(vlad_cam_path, 'vlad_nf6_' + fname), vlad_nf6)
    #
    #                         pred = self.svc_model.predict(vlad.reshape(1, -1))[0]
    #                         self.predicted.append(pred)
    #                         # print('prediction: ' , pred)
    #
    #                         color = var.green if pred == 0 else var.red
    #                         msg0 = 'L:' + str(y)
    #                         cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[1], bbox[3]), var.black, 2)
    #                         txtSize, baseLine = cv2.getTextSize(msg0, var.FONT_FACE, var.FONT_SCALE, var.thickness)
    #                         cv2.rectangle(dp, (bbox[0], bbox[2]), (bbox[0] + txtSize[0], bbox[2] + txtSize[1]), color,
    #                                       -1)
    #                         cv2.putText(dp, msg0, (bbox[0], bbox[2] + txtSize[1]), cv2.FONT_HERSHEY_COMPLEX,
    #                                     var.FONT_SCALE,
    #                                     var.white, var.thickness)
    #
    #                         msg1 = 'P:' + str(pred)
    #                         txtSize, baseLine = cv2.getTextSize(msg1, var.FONT_FACE, var.FONT_SCALE, var.thickness)
    #                         cv2.rectangle(dp, (bbox[0], bbox[2] + 20),
    #                                       (bbox[0] + txtSize[0], bbox[2] + txtSize[1] + 20), color,
    #                                       -1)
    #                         cv2.putText(dp, msg1, (bbox[0], bbox[2] + txtSize[1] + 20), cv2.FONT_HERSHEY_COMPLEX,
    #                                     var.FONT_SCALE,
    #                                     var.white, var.thickness)
    #                 end = time() - start
    #                 fps = 30 if end == 0 else 1 / end
    #                 if fps > 60:
    #                     fps = 30
    #                 cv2.putText(dp, 'fps: %d ' % (fps), (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), thickness=2)
    #
    #                 writer.write(dp)
    #
    #         # print(sample_nxt, ' has %d features '%self.count)
    #         label_fea = np.array(label_fea)
    #         abnormal_idx = np.where(label_fea==0)
    #         normal_idx = np.where(label_fea ==1)
    #         predict = np.array(self.predicted)
    #
    #         abnormal_predict = predict[abnormal_idx]
    #         abnormal_label = label_fea[abnormal_idx]
    #
    #         normal_predict = predict[normal_idx]
    #         normal_label = label_fea[normal_idx]
    #
    #         abnormal_acc = np.mean(np.equal(abnormal_predict,abnormal_label).astype(np.uint8))
    #         normal_acc = np.mean(np.equal(normal_predict,normal_label).astype(np.uint8))
    #
    #         # print('abnormal acc: %0.2f  normal acc %0.2f'%(abnormal_acc, normal_acc))
    #         path = os.path.join(var.GIST_VIOLENCE_PATH, 'predicted', fea_level + '_' + sample_nxt)
    #         pickle.dump(predict, open(path,'wb'))
    #         print('abnormal acc: %0.2f' %abnormal_acc, end=' ' , flush=True)
    #         print('normal acc: %0.2f' %normal_acc)
    #
    #
    #     print('total  pos %s , neg %s' % (format(total_pos, ','), format(total_neg, ',')))

