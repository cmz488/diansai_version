import cv2
import numpy as np

import glob

import tools


def main():
    np.set_printoptions(suppress=True)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备物体点 (9列, 6行 的内角点)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []  # 真实世界 3D 点
    imgpoints = []  # 图像平面 2D 点

    images = glob.glob("/userdata/project/photos/*.jpg")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    if gray is not None:
        h, w = gray.shape[::1]
    if objpoints is not None and imgpoints is not None and img is not None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )
    print(f"RMS:{ret}\nmtx:\n{mtx}\ndist:\n{dist}")
    if mtx is not None and dist is not None and rvecs is not None and tvecs is not None:
        np.savez("../param.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


if __name__ == "__main__":
    main()
