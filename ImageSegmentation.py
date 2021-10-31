import sys
import cv2
import random as rand

import numpy as np

import GraphOperator as go

a = set()


def generate_image(ufset, width, height):
    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(width * height)]
    #print(color)

    save_img = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            color_idx = ufset.find(y * width + x)
            #print(color_idx)
            #print("#######")
            save_img[y, x] = color[color_idx]
            a.add(color[color_idx])
    #print(a)
    return save_img

def get_roi(sigma, k , min_size, input_image_file, output_image_file):
    #sigma = float(sys.argv[1])
    #k = float(sys.argv[2])
    #min_size = float(sys.argv[3])

    img = cv2.imread(input_image_file)
    float_img = np.asarray(img, dtype=float)

    gaussian_img = cv2.GaussianBlur(float_img, (5,5), sigma)
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)

    height, width, channel = img.shape
    graph = go.build_graph(smooth_img, width, height)

    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph, key=weight)

    ufset = go.segment_graph(sorted_graph, width * height, k)
    ufset = go.remove_small_component(ufset, sorted_graph, min_size)

    save_img = generate_image(ufset, width, height)
    cv2.imwrite(output_image_file, save_img)
    print(type(save_img))
    return(a)


