"""
This file contains a implement of feature match and motion compensation within two images
"""

import cv2 as cv
import numpy as np
import operator


def feature_method_sift(image):
    # obtain feature by SIFT
    sift = cv.xfeatures2d_SIFT()
    sift = sift.create(nfeatures=100)
    return sift.detectAndCompute(image=image, mask=None)


def feature_method_orb(image):
    # obtain feature by ORB
    orb = cv.ORB()
    orb = orb.create()
    return orb.detectAndCompute(image=image, mask=None)


FEATURE_METHOD = {
                'SIFT': feature_method_sift,
                'ORB': feature_method_orb,
                }


def feature_match(feature_method: 'SIFT or ORB', src: 'Source image', sta: 'static image'):
    # feature match within two image
    match_res = []
    match_sort = []
    src_xy = []
    sta_xy = []

    src_keypoints, src_descriptors = FEATURE_METHOD[feature_method](src)
    sta_keypoints, sta_descriptors = FEATURE_METHOD[feature_method](sta)

    bfm = cv.BFMatcher().create()
    match_raw = bfm.match(src_descriptors, sta_descriptors)

    for i in range(match_raw.__len__()):
        match_sort.append([i, match_raw[i].distance])

    match_sort.sort(key=operator.itemgetter(1))
    # Sort match data using distance

    for i in range(10):  # number is given by experiment, and may make program unstable :(. You may need to change it when mistake occurs
        match_res.append(match_raw[match_sort[i][0]])

    match_img = cv.drawMatches(src, src_keypoints, sta, sta_keypoints, match_res, outImg=None)

    for i in range(match_res.__len__()):
        src_xy.append([src_keypoints[match_res[i].queryIdx].pt[0], src_keypoints[match_res[i].queryIdx].pt[1]])
        sta_xy.append([sta_keypoints[match_res[i].trainIdx].pt[0], sta_keypoints[match_res[i].trainIdx].pt[1]])

    src_xy = np.array(src_xy)
    sta_xy = np.array(sta_xy)

    return src_xy, sta_xy, match_img


def motion_compensation(src: 'Image must be moved to match sta Image', sta: 'Referred Image'):
    # Count motion transform matrix, which is used in Super-resolution and MAP method. To see maptv.py
    num = sta.__len__()
    m = []

    for count in range(num):
        src[count] = src[count].astype(np.uint8)
        sta[count] = sta[count].astype(np.uint8)

        src_xy, sta_xy, _ = feature_match(feature_method='SIFT', src=src[count], sta=sta[count])
        mat_cur = cv.estimateRigidTransform(src=src_xy, dst=sta_xy,fullAffine=True)
        compensated_src = cv.warpAffine(src[count], mat_cur, src[count].shape)

        y = np.matrix(compensated_src)
        x = np.matrix(src[count])

        m_cur = y * np.linalg.pinv(x)
        m.append(m_cur)

    return m
