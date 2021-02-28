from matplotlib import pyplot as plt
import numpy as np
import cv2


def show_image(images, figsize=(20, 50)):
    rows, cols = images.shape[:2]
    fig = plt.figure(figsize=figsize)
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        image = images[row][col]
        fig.add_subplot(rows, cols, i + 1).imshow(image)
    plt.show()


def makeBorder(image, bordersize):
    draw_image = image.copy()
    color = [1, 1, 1]
    draw_image = cv2.copyMakeBorder(draw_image,
                                    top=bordersize, bottom=bordersize,
                                    left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=color)
    return draw_image


def draw_point(image, points):
    draw_image = image.copy()
    H, W, C = draw_image.shape
    points = points * (W, H)
    points = points.astype(np.int32)
    for pnt in points:
        draw_image = cv2.circle(draw_image, tuple(
            pnt), int(H*0.01), (0, 1, 0), -1)
    return draw_image


def draw_arrow(image, src_points, dst_points):
    draw_image = image.copy()
    H, W, C = image.shape
    src_points = src_points * (W, H)
    src_points = src_points.astype(np.int32)
    dst_points = dst_points * (W, H)
    dst_points = dst_points.astype(np.int32)
    for src, dst in zip(src_points, dst_points):
        draw_image = cv2.arrowedLine(draw_image, tuple(src), tuple(dst),
                                     (1, 0, 0), int(H*0.01))
    return draw_image