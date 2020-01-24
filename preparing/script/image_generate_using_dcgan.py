# -*- coding: utf-8 -*- 
from models import Generator
import torch
import random
import torchvision.utils as vutils
import cv2
from models_wgan import DCGAN_G


def img_show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def coordinate_back_from_relative_to_absolute(relative_x, relative_y, relative_w, relative_h, img_width, img_height):
    """相対座標情報から絶対座標情報に変換する。相対とは、w,y,w,hが0-1に正規化されているもので、絶対とは、左上の座標をx1,y1 左下の座標をx2,y2のもの。
    なので、0-1で表現されたx, y, w, hに対して、0-width, 0~heightに直して返却する。"""
    x1 = img_width * (relative_x - (relative_w / 2))  # 左上のx座標
    y1 = img_height * (relative_y - (relative_h / 2))  # 左上のx座標
    x2 = img_width * (relative_x + (relative_w / 2))  # 左上のx座標
    y2 = img_height * (relative_y + (relative_h / 2))  # 左上のx座標
    return x1, y1, x2, y2


def generate_image(G_model, num, use_gan, gaussian_f):
    """return Tensor[3, 64, 64]"""
    # set config
    fake_dir = '../data/fake/'
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_gan == 'dcgan':
        ngpu = 1
        nz = 100
        ngf = 64
        nc = 3
        batchSize = 1
        # network定義
        netG = Generator(ngpu, nz, ngf, nc).to(device)
        # networkの学習済みモデルのロード
        netG.load_state_dict(torch.load(G_model))
    elif use_gan == 'wgan':
        ngpu = 1
        nz = 100
        ngf = 64
        nc = 3
        batchSize = 1
        target_original_size = (64, 64)
        target_resize_size = (30, 30)
        # network定義
        netG = DCGAN_G(isize=target_original_size[0], nz=nz, nc=nc, ngf=ngf, ngpu=ngpu, n_extra_layers=0).to(device)
        # networkの学習済みモデルのロード
        netG.load_state_dict(torch.load(G_model))

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)
    fake = netG(fixed_noise)  # batchsize, channel, height, widthが返却。

    # 作成したimageをsave.
    vutils.save_image(fake.detach(), '{}/fake_{}.png'.format(fake_dir, str(num)), normalize=True)

    img = cv2.imread('{}/fake_{}.png'.format(fake_dir, str(num)))
    resize_shape = (int(target_resize_size[0] * random.uniform(0.7, 1.3)), int(target_resize_size[1] * random.uniform(0.7, 1.3)))
    print(resize_shape)

    resize_image = cv2.resize(img, resize_shape)
    # out_name = '{}/resized_{}.png'.format(fake_dir, str(num))
    # cv2.imwrite(out_name, resize_image)
    return resize_image

def main():
    use_gan = 'wgan' # 'wgan' or 'dcgan'
    use_generatot = '../models/wgan/netG_epoch_4999.pth'
    train_file = '/home/shirai/Pytorch_YOLOv3/data/chest_imgs/brighttraindata.txt'
    original_size = (512, 512)
    put_centerx_range = 300
    put_centery_range = 200
    gaussian_f = True

    with open(train_file, 'r') as image_f:
        for i, line in enumerate(image_f):
            if i == 1:
                exit()
            line = line.rstrip()
            original_img = cv2.imread(line)

            # generate image(resized)
            generated_image = generate_image(use_generatot, i, use_gan, gaussian_f)
            anno_str = []
            
            with open(line.replace('images', 'labels').replace('png', 'txt'), 'r') as anno_f:
                for anno_line in anno_f:
                    anno_line = anno_line.rstrip()
                    anno_str.append(anno_line)
                    anno_vals = anno_line.split(' ')
                    center_x = float(anno_vals[1])
                    center_y = float(anno_vals[2])
                    w = float(anno_vals[3])
                    h = float(anno_vals[4])
                    x1, y1, x2, y2 = coordinate_back_from_relative_to_absolute(center_x, center_y, w, h, original_size[0], original_size[1])
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # 重ならないように、putする場所を決める。
                    for j in range(30):
                        x1_put = int(random.uniform((original_size[0] - put_centerx_range) / 2, (original_size[0] - put_centerx_range) / 2 + put_centerx_range))
                        y1_put = int(random.uniform((original_size[1] - put_centery_range) / 2, (original_size[1] - put_centery_range) / 2 + put_centery_range))
                        x2_put = x1_put + generated_image.shape[1]
                        y2_put = y1_put + generated_image.shape[0]
                        # 重なり判定
                        if -(x2_put - x1_put) < (x1_put - x1) and (x1_put - x1) < (x2 - x1) and \
                            -((y2_put - y1_put) + (y2 - y1)) < (y1_put - y1) and (y1_put - y1) < 0:
                            break
                            anno_str.append(anno_line)
                    if j != 30:
                        for param in range(10, 500, 10):
                            generated_image = cv2.GaussianBlur(generated_image, (5, 5), param)
                            original_img[y1_put: y2_put, x1_put: x2_put] = generated_image
                            put_center_x = (x1_put + x2_put) / (2 * original_size[1])
                            put_center_y = (y1_put + y2_put) / (2 * original_size[0])
                            put_w = (x2_put - x1_put) / original_size[1]
                            put_h = (y2_put - y1_put) / original_size[0]
                            anno_str.append('0 ' + ' '.join(map(str, [put_center_x, put_center_y, put_w, put_h])))
                            save_image_name = line.replace('.png', '_gan_gaussian_' + str(param) + '.png')
                            cv2.imwrite(save_image_name, original_img)

                    else:
                        print(line, " can't put gan image.")
                    """
                    save_image_name = line.replace('.png', '_gan_gaussian.png')
                    cv2.imwrite(save_image_name, original_img)
                    """
                    save_anno_name = save_image_name.replace('png', 'txt').replace('images', 'labels')
                    with open(save_anno_name, 'a') as anno_outf:
                        for anno in anno_str:
                            anno_outf.write(anno + '\n')
                    

if __name__ == '__main__':
    main()
