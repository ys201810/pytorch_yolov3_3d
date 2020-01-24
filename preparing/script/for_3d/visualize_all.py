# ファイル処理
import os
import sys
import glob
import zipfile
import io
# 画像処理
from PIL import Image
import numpy as np
# csv読み込み
import pandas as pd

def df_img_from_csv(csv_path, images_png):
    """
    csvをpandas.DataFrameとして読み込み、
    CT画像ファイル名と本来の輝度を対応付けた
    DataFrameを返す

    :param csv_path: csvファイルのパス
    :type csv_path: string
    :return: 画像パス、輝度情報が対応ついたDataFrame
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={'DICOM_windows':'win'},
                inplace=True)
    df_img = df[['File_name', 'win']]
    # winが"-175,275"のような文字列なので2つの小数にわける
    df_img = pd.concat([df_img,df_img['win'].str.split(',',expand=True)],axis=1).drop('win',axis=1)
    # 0,1という列名をA,Bに改名
    df_img.rename(columns={0:'A',1:'B'},inplace=True)
    # 一時的なseries
    tmp_s = df_img['File_name'].rename(columns={'File_name':'poyo'}) #どうせpoyoは0になる
    #print("Hoge")
    #print(type(str(tmp_s.str[:12])))
    #print(str(tmp_s.str[:12]))
    df_img = pd.concat([
            df_img, images_png + tmp_s.str[:12] + os.sep + tmp_s.str[13:] #'Images_png/004458_01_01/059.png'など
        ],axis=1)
    # 列名0(poyoにならない)をFile_pathに
    df_img.rename(columns={0:'File_path'},inplace=True)
    return df_img



def high_contrast(pil_img, win_min, win_max):
    """
    PIL.Image形式で読み込んだCT画像のコントラストを本来のものにする

    :param pil_img: CT画像
    :type pil_img: PIL.Image
    :param win_min: CT画像の本来の最小輝度
    :type win_min: int
    :param win_max: CT画像の本来の最大輝度
    :type win_max: int
    :return: 本来のコントラストになったCT画像とそのnp.array
    :rtype: tuple(PIL.Image, np.array)
    """
    # 画像をnp.arrayにして2^15引き、元のhounsfield unit値を得る
    hounsf_unit = np.array(pil_img) - 2.**15
    # uint8画像になるように正規化
    img_arr = (2**8-1)* np.minimum(1, np.maximum(0, (hounsf_unit-win_min)/(win_max-win_min) ))
    # uint8にキャスト
    img_arr = img_arr.astype(np.uint8)
    # PIL.Image化
    img = Image.fromarray(img_arr)
    return img, img_arr



def save_imgs_from_one_zip(zip_path,img_path,df_img):
    """
    zip中の画像を読み込んでハイコントラストにして保存する

    :param zip_path: zipファイルへのパス
    :type zip_path: string
    :param img_path: 画像の保存先. "data/"など
    :type img_path: string
    :param df_img: CT画像ファイルのパスとその輝度情報が対応付けられたデータフレーム
    :type df_img: pandas.DataFrame
    """
    z = zipfile.ZipFile(zip_path)
    files = z.namelist()
    for f in files:
        if f.endswith('png'):
            # PIL.Imageとして読み込み
            img = Image.open(io.BytesIO(z.read(f)))
            # データフレームからファイル名が一致する行を所得
            required_r = df_img.query('File_path == "{}"'.format(f))
            if len(required_r)!=0:
                # 輝度情報の所得
                A = required_r['A'].values
                A = float(A[0])
                B = required_r['B'].values
                B = float(B[0])
                # ファイル名の所得
                f_name = required_r['File_name'].values
                f_name = f_name[0]
                img, _ = high_contrast(img,A,B)
                img.save(img_path+f_name)


def get_prefix_and_CTval(files, df_img):
    for f in files:
        #print(f)
        # データフレームからファイル名が一致する行を所得
        #print(df_img["File_path"])
        required_r = df_img.query('File_path == "{}"'.format(f))
        if len(required_r)!=0: #{
            # 輝度情報の所得
            A = required_r['A'].values
            A = float(A[0])
            B = required_r['B'].values
            B = float(B[0])
            # ファイル名の所得
            f_name = required_r['File_name'].values
            f_name = f_name[0]
            prefix = f_name.rsplit("_",1)[0]
            return prefix, A, B


def save_imgs_in_one_folder(folder_path, img_path, df_img): #{
    folders = glob.glob(os.path.join(folder_path,"00*/"))
    for folder in folders:
        files = glob.glob(os.path.join(folder,"*.png"))
        print("length of this folder:", len(files))
        (prefix, A, B) = get_prefix_and_CTval(files, df_img)
        print(prefix)
        assert(prefix == folder.split(os.sep)[-2])
        for f in files: #{
            # PIL.Imageとして読み込み
            img = Image.open(f)
            # ファイル名の所得
            f_name = prefix + "_" + f.split(os.sep)[-1]
            img, _ = high_contrast(img, A, B)
            img.save(img_path+f_name)


def main():
    """
    スクリプトとして実行されたときの処理
    コマンド引数としてzipファイルがあるディレクトリとハイコントラスト画像の保存先とcsvのパスを渡す

    :return: 画像処理が正常終了した場合は0で、help表示のときは1
    :rtype: int
    """
    """
    args = sys.argv
    if len(args)!=4:
        print("number of argments must be 3 (not includes this file)")
    dev_dir = args[1]
    if dev_dir == "-h" or dev_dir == "--help":
        print("Usage: python3 visualize_all.py dataset_directory_path save_directory_path csv_path")
        return 1

    img_path = args[2]
    csv_path = args[3]
    """
    input_image_dir = '/home/nttd_msi_sm/rtec_CT_hayashi/dataset/imgs41-56/41/Images_png_41/'
    output_image_dir = '/home/shirai/Pytorch_YOLOv3/datasets/chest_imgs/images_test/'
    csv_path = '/home/shirai/Pytorch_YOLOv3/datasets/csv/DL_info_xyz_slicemmTF.csv'
    df_img = df_img_from_csv(csv_path, images_png=input_image_dir) # file_name, window_start, window_end, file_path
    save_imgs_in_one_folder(input_image_dir, output_image_dir, df_img)
    return 0


if __name__ == '__main__':
    # スクリプトとして実行されたときの処理
    main()
