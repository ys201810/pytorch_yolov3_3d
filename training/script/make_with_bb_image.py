# -*- coding: utf-8 -*- 
import glob
import cv2

def main():
    train_img_path = 'data/bright_images_labels/images/train/'
    val_img_path = 'data/bright_images_labels/images/validation/'
    train_label_path = 'data/bright_images_labels/labels/train/'
    val_label_path = 'data/bright_images_labels/labels/validation/'
    train_wbb_img_path = 'data/bright_images_labels/images/train_wbb/'
    val_wbb_img_path = 'data/bright_images_labels/images/val_wbb/'

    train_img_list = glob.glob(train_img_path + '/*.png')
    print(len(train_img_list))

    val_img_list = glob.glob(val_img_path + '/*.png')
    print(len(val_img_list))

    """
    img = cv2.imread('data/coco_test/COCO_train2014_000000000009.jpg')
    height = img.shape[0]
    width = img.shape[1]
    with open('data/coco_test/COCO_train2014_000000000009.txt') as inf:
        for line in inf:
            line = line.rstrip()
            vals = line.split(' ')
            label, x, y, w, h = vals[0], float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
            x = int(width * x)
            y = int(height * y)
            w = int(width * w)
            h = int(height * h)

            cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 3)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

    # for train_img in train_img_list:
    for train_img in train_img_list:
        img = cv2.imread(train_img)
        height = img.shape[0]
        width = img.shape[1]
        label_path = train_img.replace('/images/', '/labels/').replace('png', 'txt')
        with open(label_path, 'r') as labelf:
            for line in labelf:
                line = line.rstrip()
                vals = line.split(' ')
                label, y1, x1, h, w = vals[0], float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
                x = int((width * x1) - (int(width * w) / 2))
                y = int((height * y1) - (int(height * h) / 2))
                w = int(width * w)
                h = int(height * h)

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # cv2.rectangle(img, (y, x), (y + h, x + w), (255, 0, 0), 1)

        cv2.imwrite(train_wbb_img_path + train_img.split('/')[-1].split('.')[0] + '_wbb.png', img)

if __name__ == '__main__':
    main()