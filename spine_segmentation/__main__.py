import os
import argparse
import cv2
import numpy
from pathlib import Path
import logging

from spine_segmentation.image_proc import spine_coordinate_extraction
from spine_segmentation.image_proc import spine_image_segmentation

# コマンドラインオプションの指定
parser = argparse.ArgumentParser(
    prog='spine-segmentation',
    description='背表紙画像を分割')

parser.add_argument('--debug', action='store_true', help='デバッグログの表示')
parser.add_argument('--img', type=str, help='背表紙画像データ')
parser.add_argument('--out', type=str, default=os.environ['HOME'], help='分割画像アウトプット先')

args = parser.parse_args()

if __name__ == "__main__":
    # logging levelの指定
    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG

    # loggingConfigの指定
    logging.basicConfig(encoding='utf-8', level=logging_level)

    # 背表紙画像のチェック
    img_path = args.img
    logging.debug(f"画像データ : {img_path}")
    if img_path is None:
        raise Exception("背表紙画像ファイルが指定されていません")
    elif not (os.path.isfile(img_path)):
        raise FileNotFoundError(f"指定されたファイルが見つかりません : {img_path}")
    
    # 背表紙画像出力先のチェック
    out_path = args.out
    logging.debug(f"出力先データ : {out_path}")
    if not (os.path.isdir(out_path)):
        raise FileNotFoundError(f"出力先フォルダが見つかりません : {out_path}")

    logging.info("処理を開始します")

    img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
    contours_1 = spine_coordinate_extraction(img_data, 50)
    contours_2 = spine_coordinate_extraction(img_data, 240)
    spine_image_segmentation(img_data, (contours_1 + contours_2), out_path)

    logging.info("処理が完了しました")