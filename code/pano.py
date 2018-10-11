import numpy as np
import cv2
import sys
from matchers import matchers
import time

class Stitch:
    def __init__(self, args):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [each.rstrip('\r\n') for each in  fp.readlines()]
        print(filenames)
        self.images = [cv2.resize(cv2.imread(each),(480, 320)) for each in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [],None
        self.matcher_obj = matchers()
        self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d"%self.count)
        self.centerIdx = self.count/2 
        print("Center index image : %d"%self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if(i<=self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")
    
    #每次计算新的投影图大小，起始位置，增量拼接
    #其中起始位置，每次都做不同的单应性变换
    #其中图片大小，做了两次单应性变换，一次是a到b，一次是考虑了起点偏移量
    #其中增量拼接，每次把a集合图片往b上凑求H[b到a的逆]，然后在新的a集合上填充b
    def leftshift(self):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        for b in self.left_list[1:]:
            #返回从b到a的单应性变换H
            H = self.matcher_obj.match(a, b, 'left')
            print("Homography is : ",H)
            #从a到b的xh
            xh = np.linalg.inv(H)
            print("Inverse Homography :",xh)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
            #确保H的右下角元素为1
            ds = ds/ds[-1]
            print("final ds=>", ds)
            #每次起点，内积了xh，起点都是不同的！！！
            f1 = np.dot(xh, np.array([0,0,1]))
            f1 = f1/f1[-1]
            #H第三列的头两个元素，控制translation
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            #为了将图画完整，否则越界
            #wrong-->  dsize = (int(ds[0]),int(ds[1]))
            dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
            print("image dsize =>", dsize)
            tmp = cv2.warpPerspective(a, xh, dsize)
            # cv2.imshow("warped", tmp)
            # cv2.waitKey()
            #增量拼接，
            tmp[offsety:b.shape[0]+offsety, offsetx:b.shape[1]+offsetx] = b
            a = tmp

        self.leftImage = tmp

    
    def rightshift(self):
        for each in self.right_list:
            #返回从each到leftImage的单应性变换H
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            print("Homography :", H)
            #计算新的each图像大小
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz/txyz[-1]
            #新的each图像起始偏移为leftImage大小
            dsize = (int(txyz[0])+self.leftImage.shape[1], int(txyz[1])+self.leftImage.shape[0])
            print('dsize:', dsize)
            tmp = cv2.warpPerspective(each, H, dsize)
            #cv2.imshow("tp", tmp)
            #cv2.waitKey()
            # tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
            tmp = self.mix_and_match(self.leftImage, tmp)
            print("tmp shape",tmp.shape)
            print("self.leftimage shape=", self.leftImage.shape)
            self.leftImage = tmp
        # self.showImage('left')


    #每次warpedImage调用mix_and_match前，有大量的黑场
    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        print(leftImage[-1,-1])

        t = time.time()
        black_l = np.where(leftImage == np.array([0,0,0]))
        black_wi = np.where(warpedImage == np.array([0,0,0]))
        print(time.time() - t)
        print(black_l[-1])

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    #真的是空白
                    if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                        # print "BLACK"
                        # instead of just putting it with black, 
                        # take average of all nearby values and avg it.
                        warpedImage[j,i] = [0, 0, 0]
                    else:
                        #用left image填补空白
                        if(np.array_equal(warpedImage[j,i],[0,0,0])):
                            # print "PIXEL"
                            warpedImage[j,i] = leftImage[j,i]
                        else:
                            #融合
                            if not np.array_equal(leftImage[j,i], [0,0,0]):
                                bw, gw, rw = warpedImage[j,i]
                                bl,gl,rl = leftImage[j,i]
                                # b = (bl+bw)/2
                                # g = (gl+gw)/2
                                # r = (rl+rw)/2
                                warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage




    def trim_left(self):
        pass

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
            # cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()


if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "txtlists/files3.txt"
    finally:
        print("Parameters : ", args)
    s = Stitch(args)
    s.leftshift()
    # s.showImage('left')
    s.rightshift()
    print("done")
    cv2.imwrite("test_file3.jpg", s.leftImage)
    print("image written")
    #cv2.destroyAllWindows()
    
