import sys
import os
import urllib.request
import tarfile
import zipfile
import numpy as np

class Cifar_Downloader():
    def __init__(self):
        self.pre_pct_complete = -1
        print("=====(주의) numpy가 설치되어 있어야 합니다.=====")
        data_path = "data/CIFAR-10/"
        data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.maybe_download_and_extract(url=data_url, download_dir=data_path)
        self.run()


    def _print_download_progress(self,count, block_size, total_size):

        pct_complete = float(count * block_size) / total_size
        pct_complete = round(pct_complete,3)
        if self.pre_pct_complete != pct_complete:
            msg = "\r- 다운로드 진행 중 : {0:.1%}\n".format(pct_complete)
            self.pre_pct_complete = pct_complete
            sys.stdout.write(msg)
            sys.stdout.flush()

    def maybe_download_and_extract(self,url, download_dir):
        filename = url.split('/')[-1]
        file_path = os.path.join(download_dir, filename)

        if not os.path.exists(file_path):
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            file_path, _ = urllib.request.urlretrieve(url=url,filename=file_path,reporthook=self._print_download_progress)

            print()
            print("다운로드가 완료되었습니다. 데이터를 추출합니다.\n")

            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

            print("완료.\n")
        else:
            os.remove(file_path)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            file_path, _ = urllib.request.urlretrieve(url=url,filename=file_path,reporthook=self._print_download_progress)

            print()
            print("다운로드가 완료되었습니다. 데이터를 추출합니다.\n")

            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

            print("완료.\n")


    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def run(self):
        name = 'data_batch_1'
        batch = self.unpickle("data/CIFAR-10/cifar-10-batches-py/{}".format(name))
        trainX = batch[b'data']
        trainY = batch[b'labels']
        for name in ['data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
            batch = self.unpickle("data/CIFAR-10/cifar-10-batches-py/{}".format(name))
            trainX = np.append(trainX,batch[b'data'],axis=0)
            trainY = np.append(trainY,batch[b'labels'],axis=0)
        #dict_keys([b'filenames', b'labels', b'batch_label', b'data'])

        test = self.unpickle("data/CIFAR-10/cifar-10-batches-py/test_batch")
        testX = test[b'data']
        testY = test[b'labels']
        print("npy로 변환을 시작합니다.\n")
        #dict_keys([b'filenames', b'labels', b'batch_label', b'data'])

        trainY = np.expand_dims(trainY,axis=1)
        testY = np.expand_dims(testY,axis=1)

        t_1 = []
        t_2 = []

        for i in range(50000):
            if trainY[i] == 0:
                t_1.append(i)
            elif trainY[i] == 1:
                t_1.append(i)
            elif trainY[i] == 2:
                t_1.append(i)

        for i in range(10000):
            if testY[i] == 0:
                t_2.append(i)
            elif testY[i] == 1:
                t_2.append(i)
            elif testY[i] == 2:
                t_2.append(i)

        trainX = trainX[t_1,:]
        trainY = trainY[t_1]
        testX = testX[t_2,:]
        testY = testY[t_2,:]

        np.save('trainX',trainX)
        np.save('trainY',trainY)
        np.save('testX',testX)
        np.save('testY',testY)


        print("종료합니다.")

Cifar_Downloader()

    


