import os
import cv2
import numpy
from pathlib import Path
import logging


def spine_coordinate_extraction(
    image_data: numpy.ndarray,
    bw_thresh: int = 50
) -> tuple:
    """Summary line.

    背表紙画像から座標を抽出してタプルで返す

    Parameters
    ----------
    src: numpy.ndarray
        背表紙画像データ
    thresh: int
        2値化する際のしきい値(デフォルト値 : 50)

    Returns
    -------
    tuple
        背表紙座標データ
    """
    # グレイスケールに変換
    grayscale_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # 2値化
    ret_val, bw_data = cv2.threshold(
        grayscale_data, bw_thresh, 255, cv2.THRESH_BINARY)

    # 画像データから輪郭(背表紙)を抽出
    contours, hierarchy = cv2.findContours(
        bw_data, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours


def spine_image_segmentation(
        image_data: numpy.ndarray,
        contours: tuple,
        spin_image_path: str = os.environ['HOME'],
        spines_max_size: int = 50000,
        spines_min_size: int = 15000
):
    """Summary line.

    背表紙座標から

    Parameters
    ----------
    src: str
        背表紙画像データ
    thresh: int
        2値化する際のしきい値(デフォルト値 : 50)

    Returns
    -------
    tuple
        背表紙座標データ
    """
    # 検出された背表紙数の初期化
    spines_count = 0

    for i in range(len(contours)):
        # 輪郭の領域を計算
        area = cv2.contourArea(contours[i])

        # 抽出した範囲がサイズ範囲内かつ座標データが存在
        if (spines_min_size < area and area < spines_max_size) and len(contours[i]):
            # 四隅を抽出
            x, y, w, h = cv2.boundingRect(contours[i])

            # 背表紙画像を保存
            spine_output_path = str(
                Path(spin_image_path, str(spines_count) + '.jpg'))
            cv2.imwrite(spine_output_path, image_data[y:y + h, x:x + w])

            logging.debug(f"save spine : {spine_output_path}")

            # 抽出数をインクリメント
            spines_count = spines_count + 1

    logging.info(f"こちらに背表紙画像が格納されました : {spin_image_path}")


if __name__ == "__main__":
    # デバッグ用
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    img_path = ""
    img_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
    contours_1 = spine_coordinate_extraction(img_data, 50)
    contours_2 = spine_coordinate_extraction(img_data, 240)
    spine_image_segmentation(img_data, (contours_1 + contours_2))
