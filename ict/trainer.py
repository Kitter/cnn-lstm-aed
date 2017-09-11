import os

from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


from ict.read_txt import *
import logging, logging.handlers

from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class trainer:
    def __init__(self, hps, vgg, sess):
        self.hps = hps
        self.sess = sess
        self.vgg = vgg
        vgg_level = 'vgg' + str(hps.cnn_ver)
        clip_level = 'clip'
        pca_level =  clip_level+str(hps.clip_n)+'_pca' + str(hps.pca_k)
        vlad_level = pca_level + '_vlad' + str(hps.vlad_k)

        vgg_bpath = fm.mkdir(os.path.join(var.GIST_VIOLENCE_PATH, vgg_level))
        self.clip_bpath = os.path.join(vgg_bpath, clip_level)
        self.pca_bpath = fm.mkdir(os.path.join(vgg_bpath,pca_level))

        self.vlad_bpath = fm.mkdir(os.path.join(vgg_bpath, vlad_level))
        self.vlad_maker = VLAD(self.hps)
        self.init_logger()



    def init_logger(self):

        # logger 인스턴스를 생성 및 로그 레벨 설정
        self.logger = logging.getLogger("crumbs")
        self.logger.setLevel(logging.DEBUG)
        #  formmater 생성
        formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        # fileHandler와 StreamHandler를 생성

        fileHandler = logging.FileHandler(var.PROJECT_PATH+'/my.log')
        streamHandler = logging.StreamHandler()

        #  handler에 fommater 세팅
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        #  Handler를 logging에 추가
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

        file_max_bytes = 10 * 1024 * 1024 # 10MB
        # if file exceeds 10MB delete oldest one and remains last 10MB log.
        # fileHandler = logging.handlers.RotatingFileHandler(filename='./log/test.log', maxBytes=file_max_bytes,
        #                                                    backupCount=10)



    def descriptor_idx(self, i, k):
        return str(i).zfill(5) + '_' + str(k).zfill(2) + '.npy'

    def descriptor_name(self,idx,sig):
        return sig+'_'+idx

    def make_cnn_feature22(self):

        sample_list = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']
        src_path = var.GIST_VIOLENCE_PATH

        print('making train-set to', self.clip_bpath)
        for sample_nxt in sample_list:  # for each video
            print(sample_nxt, end=' ', flush=True)

            clip_cam_path = fm.mkdir(os.path.join(self.clip_bpath, sample_nxt))

            if os.path.exists(clip_cam_path + '/' + var.KEYS):
                continue

            if var.DEBUG:
                print('Writing at ', clip_cam_path)

            bboxX = get_bbox(src_path, sample_nxt)  # bounding-box of video
            labelX = frame_label(sample_nxt)

            feaX, feaY = [], []

            keys = list(bboxX.keys())
            n_frame = 13000
            cap, props = im.read_video_by_path(src_path, sample_nxt + '.avi')
            i = -1
            abnormal_count, normal_count = 0, 0

            feature_set = {}
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
                        if label == 0:
                            normal_count += 1
                        else:
                            abnormal_count+= 1

                        fea_idx = self.descriptor_idx(i, k)
                        fname = self.descriptor_name(fea_idx,'fc6')

                        x1, x2, y1, y2 = bbox[k]
                        height = y2 - y1
                        height = height * 1.1
                        height2 = int(height / 2)
                        cx = int((x1 + x2) / 2)

                        x1_ = cx - height2
                        x2_ = cx + height2

                        x1_ = self.not_zero(x1_)
                        x2_ = self.not_zero(x2_)

                        roi = cv2.resize(frame[y1:y2, x1_:x2_], (224, 224))
                        f6 = self.sess.run(self.vgg.fc1, feed_dict={self.vgg.imgs: [roi]})
                        # nf6 = normalize(f6, axis=0)  # default is l2
                        # feature_set[fname] = nf6[0]
                        feature_set[fname] = f6[0]
                        feaX.append(fea_idx)
                        feaY.append(label)


            cap.release()
            feaY = np.array(feaY)
            pickle.dump(feature_set, open(clip_cam_path + '/'+var.FEATURE, 'wb'))
            pickle.dump(feaX, open(clip_cam_path + '/'+var.KEYS, 'wb'))
            pickle.dump(feaY, open(clip_cam_path + '/'+var.LABEL, 'wb'))

            print(sample_nxt, ' abnormal %s + normal %s' % (format(abnormal_count, ','), format(normal_count, ',')))
        print('Done')

    def not_zero(self, x):
        if x < 0:
            return 0
        else:
            return x

    def train_pca(self):
        from sklearn.decomposition import PCA

        model_path = os.path.join(self.pca_bpath, 'pca_model'+str(self.hps.pca_k))
        if os.path.exists(model_path):
            pca = pickle.load(open(model_path, 'rb'))
        else:
            pca = PCA(n_components=512)
            X, Y = self.load_trainset(sample_list, self.clip_bpath,'fc6')
            pca.fit(X, Y)
            pickle.dump(pca, open(model_path, 'wb'))

        return pca

    def dimension_reduction(self):

        pca = self.train_pca()

        print('PCA reduction: ')

        for sample_nxt in sample_list:
            print(sample_nxt, end=' ', flush=True)
            pca_path = fm.mkdir(os.path.join(self.pca_bpath, sample_nxt))
            if os.path.exists(pca_path + '/'+var.KEYS):
                continue


            clip_path = fm.mkdir(os.path.join(self.clip_bpath, sample_nxt))
            namex = pickle.load(open(clip_path + '/' + var.KEYS, 'rb'))
            namey = pickle.load(open(clip_path + '/' + var.LABEL, 'rb'))
            cnn_feature = pickle.load(open(clip_path + '/' + var.FEATURE, 'rb'))

            X_=[]
            for x in namex:
                inx = self.descriptor_name(x,'fc6')
                fx = cnn_feature[inx]
                X_.append(fx)



            pca_set = {}
            pcax = pca.transform(X_)
            for fx, idx in zip(pcax, namex):
                xx = self.descriptor_name(idx,'pca')
                pca_set[xx] = fx

            pickle.dump(pca_set, open(pca_path + '/' + var.FEATURE, 'wb'))
            pickle.dump(namex, open(pca_path + '/' + var.KEYS, 'wb'))
            pickle.dump(namey, open(pca_path + '/' + var.LABEL, 'wb'))



        print('Done')



    def encoding(self):

        # fisher_maker = FISHER()
        # print('Encoding at', self.clip_bpath, ' >> ' , self.vlad_bpath)

        print('Encoding ')
        self.vlad_maker.ready_for()
        clip = self.hps.clip_n - 1


        for sample_nxt in sample_list:

            print(sample_nxt, end=' ', flush=True)
            vlad_path = fm.mkdir(os.path.join(self.vlad_bpath, sample_nxt))
            if os.path.exists(vlad_path + '/'+var.KEYS):
                continue

            pca_path = fm.mkdir(os.path.join(self.pca_bpath, sample_nxt))
            bboxX = get_bbox(var.GIST_VIOLENCE_PATH, sample_nxt)
            keys = list(bboxX.keys())
            fea_dic = {}
            labelX = frame_label(sample_nxt)
            namex , namey = [], []

            pca_feature = pickle.load(open(pca_path+'/'+var.FEATURE,'rb'))

            vlad_set = {}
            for j in range(13000): # current idx

                if var.DEBUG:
                    tools.progress_bar(j,13000,10)

                if j in keys: # if current frame has bounding-box
                    frame_dic = {}
                    curr_bbox = bboxX[j]
                    for k in curr_bbox.keys(): # load each person.
                        idx = self.descriptor_idx(j,k)
                        fxname = self.descriptor_name(idx,'pca')
                        fx = pca_feature[fxname]
                        frame_dic[k] = fx

                    fea_dic[j] = frame_dic

                    i = j - clip  # previous idx
                    label = labelX[j]
                    if i in keys: # if previous frame has bbox
                        prev_bbox = bboxX[i]

                        for k in curr_bbox.keys():
                            if k in prev_bbox.keys():

                                to_vlad = []
                                for w in range(i,j+1): # between prev and current
                                    if not w in fea_dic.keys():
                                        print('frame %d has no bbox'%w)
                                        continue

                                    if k in fea_dic[w].keys():
                                        fff = fea_dic[w][k]
                                        to_vlad.append(fff)

                                vlad = self.vlad_maker.get_vlad_descriptor(to_vlad)
                                if len(to_vlad) < self.hps.clip_n - 5:
                                    print('frame %d obj %d has only %d fx' % (i, k, len(to_vlad)))
                                    continue

                                idx = self.descriptor_idx(i, k)
                                fxname = self.descriptor_name(idx, 'vlad')
                                vlad_set[fxname] = vlad
                                namex.append(idx)
                                namey.append(label)

            pickle.dump(vlad_set, open(vlad_path + '/' + var.FEATURE, 'wb'))
            pickle.dump(namex, open(vlad_path + '/' + var.KEYS, 'wb'))
            pickle.dump(namey, open(vlad_path + '/' + var.LABEL, 'wb'))


        print('Done')



    def train_kmeans(self):

        if not self.vlad_maker.exits():
            X, Y = [], []

            for sample_nxt in sample_list:
                print(sample_nxt, end=' ', flush=True)
                pca_path = fm.mkdir(os.path.join(self.pca_bpath, sample_nxt))

                feax = pickle.load(open(pca_path + '/'+var.FEATURE, 'rb'))
                namex = pickle.load(open(pca_path + '/'+var.KEYS, 'rb'))
                namey = pickle.load(open(pca_path + '/'+var.LABEL, 'rb'))


                for x, y in zip(namex, namey):
                    xx = self.descriptor_name(x,'pca')
                    fx = feax[xx]
                    X.append(fx)
                    Y.append(y)

            X = np.vstack(X)
            Y = np.array(Y)
            self.vlad_maker.train_kmeans(X, Y)
            print('Done')

    def train_svc(self, k):


        model_path = os.path.join(self.vlad_bpath,'svc_model'+str(k))
        if var.DEBUG:
            print('Loading model: ', model_path)

        if not os.path.exists(model_path):

            if k == -1:
                trainX, trainY = self.load_trainset(sample_list, self.vlad_bpath,'vlad')
            else:
                sample_list_ = []
                for i in range(len(sample_list)):
                    if i != k:
                        sample_list_.append(sample_list[i])

                trainX, trainY = self.load_trainset(sample_list_, self.vlad_bpath, 'vlad')

            print('svc training', end=' ', flush=True)
            svc = LinearSVC()
            svc.fit(trainX, trainY)
            pickle.dump(svc, open(model_path, 'wb'))
            print('Done')

        return pickle.load(open(model_path, 'rb'))

    def load_trainset(self, sample_list, given_path, sig):
        X, Y = [], []
        print('loading trainset: ', end=' ', flush=True)
        if var.DEBUG:
            print('from ', given_path)

        for sample_nxt in sample_list:

            print(sample_nxt, end=' ', flush=True)
            tmp_path = fm.mkdir(os.path.join(given_path, sample_nxt))

            feax = pickle.load(open(tmp_path + '/' + var.FEATURE, 'rb'))
            namex = pickle.load(open(tmp_path + '/' + var.KEYS, 'rb'))
            namey = pickle.load(open(tmp_path + '/' + var.LABEL, 'rb'))

            for x, y in zip(namex, namey):
                xx =self.descriptor_name(x,sig)
                fx = feax[xx]
                X.append(fx)
                Y.append(y)

        print('Done')
        return X, Y



    def recall(self, predicted, label):

        positive_idx = np.where(label == 1)[0]
        positives = predicted[positive_idx]  # domain is label ==1
        tp = np.count_nonzero(positives == 1)
        fn = np.count_nonzero(positives == 0)
        b = tp + fn

        recall = tp / b if b != 0 else 0
        return recall

    def precision(self, predict, label):
        pred_pos_idx = np.where(predict == 1)
        pred_pos = label[pred_pos_idx]

        tp = np.count_nonzero(pred_pos == 1)
        fp = np.count_nonzero(pred_pos == 0)
        b = tp + fp
        precision = tp / b if b != 0 else 0
        return precision




    def cross_validation(self):
        self.logger.debug(self.hps)
        self.logger.debug('Cross validation')

        for k ,sample_nxt in enumerate(sample_list):
            model = self.train_svc(k)
            testX, testY = self.load_trainset([sample_nxt], self.vlad_bpath, 'vlad')
            label_fea = np.array(testY)
            predict = model.predict(testX)
            acc = self.accuracy(predict, label_fea)
            self.logger.debug('Test on %s, %s'%(sample_nxt, acc)  )
        self.logger.debug('')


    def accuracy(self, predict, label):

        abnormal_idx = np.where(label == 1)
        normal_idx = np.where(label == 0)
        predict = np.array(predict)

        abnormal_predict = predict[abnormal_idx]
        abnormal_label = label[abnormal_idx]

        normal_predict = predict[normal_idx]
        normal_label = label[normal_idx]
        if len(abnormal_predict) == 0:
            abnormal_acc = -1
        else:
            abnormal_acc = np.mean(np.equal(abnormal_predict, abnormal_label).astype(np.uint8))

        normal_acc = np.mean(np.equal(normal_predict, normal_label).astype(np.uint8))
        mean_acc = np.mean(np.equal(predict, label).astype(np.uint8))
        # print('abnormal acc: %0.2f  normal acc %0.2f'%(abnormal_acc, normal_acc))



        # print('abnormal acc: %0.2f' % abnormal_acc, end=' ', flush=True)
        # print('normal acc: %0.2f' % normal_acc, end=' ', flush=True)
        # print('mean acc:%0.2f ' % mean_acc)
        # print()

        return 'abnormal acc: %0.2f, normal acc %0.2f'%(abnormal_acc, normal_acc)



    def plotting_each_cam(self,sig):


        # if sig=='fc6':
        #     X,Y = self.load_trainset(sample_list, self.clip_bpath, sig)
        # else:
        #     X,Y = self.load_trainset(sample_list, self.vlad_bpath, sig)
        # pca = PCA(n_components=2)
        # pca.fit(X,Y)
        count = 0
        for sample in sample_list:

            if sig == 'fc6':
                X, Y = self.load_trainset([sample], self.clip_bpath, sig)
            else:
                X, Y = self.load_trainset([sample], self.vlad_bpath, sig)

            pca = PCA(n_components=2)
            X_ = pca.fit_transform(X, Y)
            X_ = pca.transform(X)
            print('training DONE')

            count+= 1
            ax = plt.subplot(2,3,count)

            X_ = np.array(X_)
            Y = np.array(Y)
            abnormal_idx = np.where(Y==1)
            abnormal_x = X_[abnormal_idx]
            normal_idx = np.where(Y==0)
            normal_x= X_[normal_idx]

            ax.plot(normal_x[:,0], normal_x[:,1], 'go')
            ax.plot(abnormal_x[:,0], abnormal_x[:,1], 'ro')
            plt.title(sample)

            red_patch = mpatches.Patch(color='red', label='The red data')
            green_patch = mpatches.Patch(color='green', label='The red data')
            plt.legend(handles=[red_patch,green_patch])

        # plt.tight_layout()
        # t = str(self.hps)
        # plt.savefig('f:/tmp/'+t+'_'+sig+'.png')
        plt.show()
        plt.clf()






    def plotting(self, sig):

        if sig=='fc6':
            X,Y = self.load_trainset(sample_list, self.clip_bpath, sig)
        else:
            X,Y = self.load_trainset(sample_list, self.vlad_bpath, sig)

        pca = PCA(n_components=2)
        X_ = pca.fit_transform(X,Y)
        print('training DONE')


        X_ = np.array(X_)
        Y = np.array(Y)
        abnormal_idx = np.where(Y==1)
        abnormal_x = X_[abnormal_idx]
        normal_idx = np.where(Y==0)
        normal_x= X_[normal_idx]

        plt.plot(normal_x[:,0], normal_x[:,1], 'go')
        plt.plot(abnormal_x[:,0], abnormal_x[:,1], 'ro')

        # for x in abnormal_x:
        #     plt.plot(x[0], x[1], 'ro')
        #
        # for x in normal_x:
        #     plt.plot(x[0], x[1], 'go')

        red_patch = mpatches.Patch(color='red', label='The red data')
        green_patch = mpatches.Patch(color='green', label='The red data')
        plt.legend(handles=[red_patch,green_patch])
        plt.title(self.hps)
        plt.tight_layout()
        t = str(self.hps)
        plt.savefig('f:/tmp/'+t+'_'+sig+'.png')
        plt.clf()





    def experiment(self):


        model = self.train_svc(k = -1)

        for sample_nxt in sample_list:
            testX, testY = self.load_trainset([sample_nxt], self.vlad_bpath)
            label_fea = np.array(testY)
            predict = model.predict(testX)
            self.accuracy(predict,label_fea)


