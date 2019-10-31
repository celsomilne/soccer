import numpy as np
import cv2
import pandas as pd

from ._base import ModelBase


def merge_similar_lines(l, lines):
    if len(lines.shape) == 1:
        lines = np.array([lines])

    # Translate to similar line bounds
    d = np.column_stack(
        ((lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2)
    )
    d = np.tile(d, (1, 2))
    tl = l - d

    # Rotate line to similar line bounds
    xd = lines[:, 2] - lines[:, 0]
    yd = lines[:, 3] - lines[:, 1]
    td = np.sqrt(xd * xd + yd * yd)
    cos_theta = xd / td
    sin_theta = yd / td
    tl = np.column_stack(
        (
            tl[:, 0] * cos_theta + tl[:, 1] * sin_theta,
            tl[:, 1] * cos_theta - tl[:, 0] * sin_theta,
            tl[:, 2] * cos_theta + tl[:, 3] * sin_theta,
            tl[:, 3] * cos_theta - tl[:, 2] * sin_theta,
        )
    )

    # Bounds for the lines to be considered similar
    xb = (
        np.sqrt((lines[:, 0] - lines[:, 2]) ** 2 + (lines[:, 1] - lines[:, 3]) ** 2) / 2
        + 10
    )
    yb = 15

    similar = np.logical_and(abs(tl[:, 1]) < yb, abs(tl[:, 3]) < yb)
    if sum(similar) > 1:
        diffs = np.maximum(abs(tl[:, 1]), abs(tl[:, 3]))
        similar[:] = False
        similar[np.argmin(diffs)] = True

    if any(similar):
        xb = xb[similar]
        if tl[similar, 0] < -xb or tl[similar, 0] > xb:
            lines[similar, 0:2] = l[0:2]
        if tl[similar, 2] < -xb or tl[similar, 2] > xb:
            lines[similar, 2:4] = l[2:4]
    else:
        lines = np.concatenate((lines, l.reshape(1, 4)), axis=0)

    return lines


def is_edge_line(l, dims):
    tol = 10
    bad = (
        (l[0] < tol and l[2] < tol)
        or (l[1] < tol and l[3] < tol)
        or (l[0] > dims[1] - tol and l[2] > dims[1] - tol)
        or (l[1] > dims[0] - tol and l[3] > dims[0] - tol)
    )
    return bad


def find_intersection(l1, l2):
    m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
    b1 = l1[1] - m1 * l1[0]

    if l2[0] == l2[2]:
        return np.array([l2[0], m1 * l2[0] + b1])

    m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
    b2 = l2[1] - m2 * l2[0]

    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return np.array([x, y])


def generate_points(line_dict, right_view, true_dict, rel_dict):

    A = np.array([]).reshape(0, 8)
    b = np.array([])
    for n1 in rel_dict.keys():
        if n1 in line_dict:
            for n2 in rel_dict[n1]:
                if n2 in line_dict:
                    x = np.array([0.0, 0.0])
                    x += true_dict[n1]
                    if right_view:
                        x += true_dict[n2]
                    else:
                        x -= true_dict[n2]
                    u = find_intersection(line_dict[n1], line_dict[n2])
                    A = np.vstack(
                        (
                            A,
                            [
                                [u[0], u[1], 1, 0, 0, 0, -u[0] * x[0], -u[1] * x[0]],
                                [0, 0, 0, u[0], u[1], 1, -u[0] * x[1], -u[1] * x[1]],
                            ],
                        )
                    )
                    b = np.concatenate((b, x))

    if right_view == 1:
        u = find_intersection(line_dict["top"], line_dict["left"])
        A = np.vstack((A, [[0, 0, 0, u[0], u[1], 1, -70 * u[0], -70 * u[1]]]))
        u = find_intersection(line_dict["top"], line_dict["right"])
        A = np.vstack((A, [[0, 0, 0, u[0], u[1], 1, -70 * u[0], -70 * u[1]]]))
        b = np.concatenate((b, [70, 70]))
        """
    elif right_view == 0:
        u = find_intersection(line_dict['top'],line_dict['right'])
        A = np.vstack((A,[[0,0,0,u[0],u[1],1,-70*u[0],-70*u[1]]]))
        u = find_intersection(line_dict['goal'],line_dict['left'])
        A = np.vstack((A,[[u[0],u[1],1,0,0,0,50*u[0],50*u[1]])
        b = np.concatenate((b,[70,-50]))
    elif right_view == 2:
        u = find_intersection(line_dict['top'],line_dict['left'])
        A = np.vstack((A,[[0,0,0,u[0],u[1],1,-70*u[0],-70*u[1]]]))
        u = find_intersection(line_dict['goal'],line_dict['right'])
        A = np.vstack((A,[[u[0],u[1],1,0,0,0,-50*u[0],-50*u[1]])
        b = np.concatenate((b,[70,50]))
        """

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c = np.concatenate((c, np.array([1.0])))
    c = c.reshape(3, 3)

    return c


class FieldTransform:
    true_dict = {
        "top": np.array([0.0, 70.0]).astype(np.float32),
        "mid": np.array([0.0, 0.0]).astype(np.float32),
        "box": np.array([33.5, 0.0]).astype(np.float32),
        "tbox": np.array([0.0, 55.15]).astype(np.float32),
        "bbox": np.array([0.0, 14.85]).astype(np.float32),
        "goal": np.array([50.0, 0.0]).astype(np.float32),
        "tcirc": np.array([0.0, 39.575]).astype(np.float32),
        "bcirc": np.array([0.0, 30.425]).astype(np.float32),
        "rcirc": np.array([4.575, 0.0]).astype(np.float32),
        "lcirc": np.array([-4.575, 0.0]).astype(np.float32),
        "ccirc": np.array([0.0, 35.0]).astype(np.float32),
    }

    rel_dict = {
        "top": ["mid", "box", "goal"],
        "tbox": ["box", "goal"],
        "bbox": ["box", "goal"],
        "tcirc": ["mid"],
        "bcirc": ["mid"],
        "ccirc": ["lcirc", "rcirc"],
    }

    def __call__(self, im):
        mask_field = np.logical_and(
            im[:, :, 1] > im[:, :, 0], im[:, :, 1] > im[:, :, 2]
        ).astype(np.uint8)

        k = 40
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask_field, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2 * k, 2 * k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        gray = im[:, :, 2]

        low_threshold = 30
        high_threshold = 200
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.dilate(edges, np.ones((3, 3)))

        rho_tol = 1
        theta_tol = np.pi / 180
        lines = np.array(
            cv2.HoughLinesP(
                edges, rho_tol, theta_tol, 50, minLineLength=100, maxLineGap=5
            )
        )
        lines.shape = (lines.shape[0], lines.shape[2])

        good_lines = np.array([l for l in lines if not is_edge_line(l, edges.shape)])

        lines = good_lines[0]
        for i in range(1, len(good_lines)):
            lines = merge_similar_lines(good_lines[i], lines)

        theta = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]) % np.pi

        line_dict = {
            "left": np.array([0, 0, 0, edges.shape[0]]),
            "right": np.array([edges.shape[1], 0, edges.shape[1], edges.shape[0]]),
        }
        right_view = 2

        if np.any(abs(theta - np.pi / 2) < np.pi / 60) or len(theta) < 2:
            right_view = 1
            line_dict["mid"] = lines[np.argmin(abs(theta - np.pi / 2))]
            line_dict["top"] = lines[np.pi / 2 - abs(theta - np.pi / 2) < np.pi / 30][0]
            midpoint = (line_dict["mid"][0] + line_dict["mid"][2]) // 2

            line_mask = np.ones(gray.shape, np.uint8)
            for x1, y1, x2, y2 in line_dict.values():
                cv2.line(line_mask, (x1, y1), (x2, y2), (0, 0, 0), 10)
            line_mask[: line_mask.shape[0] // 8, :] = 0
            line_mask[
                (line_mask.shape[0] - line_mask.shape[0] // 8) : line_mask.shape[0], :
            ] = 0
            line_mask[:, :20] = 0
            line_mask[:, line_mask.shape[1] - 20 :] = 0

            edges = cv2.bitwise_and(edges, edges, mask=line_mask)
            edges = cv2.dilate(edges, np.ones((3, 3)))

            _, conts, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            c = max(conts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            line_dict["tcirc"] = np.array([x, y, x + w, y])
            line_dict["bcirc"] = np.array([x, y + h, x + w, y + h])
            line_dict["ccirc"] = np.array([x, y + h // 2, x + w, y + h // 2])
            if x < midpoint:
                line_dict["lcirc"] = np.array([x, y, x, y + h])
            else:
                line_dict["rcirc"] = np.array([x + w, y, x + w, y + h])
        else:
            horz_theta = theta[np.argmin(np.pi / 2 - abs(theta - np.pi / 2))]

            if horz_theta < np.pi / 2:
                right_view = 0

            horz_lines = lines[abs(theta - horz_theta) < np.pi / 30]
            line_dict["top"] = horz_lines[
                np.argmin(np.minimum(horz_lines[:, 1], horz_lines[:, 3]))
            ]
            lines = lines[np.any(lines != line_dict["top"], axis=1)]
            horz_lines = horz_lines[np.any(horz_lines != line_dict["top"], axis=1)]

            if right_view:
                goal_line_check = np.logical_and(
                    lines[:, 0] - line_dict["top"][2] < 100,
                    lines[:, 1] - line_dict["top"][3] < 20,
                )
            else:
                goal_line_check = np.logical_and(
                    lines[:, 2] - line_dict["top"][0] < 100,
                    lines[:, 3] - line_dict["top"][1] < 20,
                )

            if any(goal_line_check):
                line_dict["goal"] = lines[goal_line_check][0]
                lines = lines[np.logical_not(goal_line_check)]

            line_dict["tbox"] = horz_lines[
                np.argmin(horz_lines[:, 1] + horz_lines[:, 3])
            ]
            lines = lines[np.any(lines != line_dict["tbox"], axis=1)]
            horz_lines = horz_lines[np.any(horz_lines != line_dict["tbox"], axis=1)]

            if len(horz_lines) != 0 and "tbox" in line_dict:
                bbox_temp = horz_lines[np.argmin(horz_lines[:, 1] + horz_lines[:, 3])]
                if np.max(bbox_temp[[1, 3]]) - np.max(line_dict["tbox"][[1, 3]]) > 2 * (
                    np.max(line_dict["tbox"][[1, 3]]) - np.max(line_dict["top"][[1, 3]])
                ):
                    line_dict["bbox"] = bbox_temp
                    lines = lines[np.any(lines != line_dict["bbox"], axis=1)]
                    horz_lines = horz_lines[
                        np.any(horz_lines != line_dict["bbox"], axis=1)
                    ]

            if right_view:
                box_line_check = (
                    np.linalg.norm(lines[:, 0:2] - line_dict["tbox"][0:2], axis=1) < 50
                )
            else:
                box_line_check = (
                    np.linalg.norm(lines[:, 2:4] - line_dict["tbox"][2:4], axis=1) < 50
                )
            if any(box_line_check):
                line_dict["box"] = lines[box_line_check][0]
                lines = lines[np.logical_not(box_line_check)]

        c = generate_points(line_dict, right_view, self.true_dict, self.rel_dict)

        return c


class SoccerModel:
    ft = FieldTransform()
    obj_old = pd.DataFrame()

    def __init__(self, detector):
        self.detector = detector
        self.objects = detector.bb_df

    def predict(self, frameIdx):

        # Pull objects in current dataframe
        obj = self.detector.get_df(batchNum=frameIdx)
        frame = self.get_frame(obj)

        width = obj["width"]
        height = obj["height"]

        obj = obj[np.abs(width - np.mean(width)) / np.std(width) < 2]
        obj = obj[np.abs(height - np.mean(height)) / np.std(height) < 2]
        obj["ufeet"] = obj["left"] + obj["width"] / 2
        obj["vfeet"] = obj["top"] + obj["height"]

        c = ft(frame)

        u = np.vstack((obj["ufeet"].values, obj["vfeet"].values, np.ones((1, len(df)))))
        x = c.dot(u).T
        x = x / x[:, 2]

        x_old = self.old_obj.loc[:, ["xnew", "ynew"]].values
        obj.loc[:, ["xnew", "ynew"]] = x[:, 0:2]

        if frame == 1:  # or frame drastically changed:
            obj["vel"] = np.ones((1, len(obj))) * np.nan
            self.obj_old = obj
        else:
            diffs = np.stack((x[:, 0] - x_old[:, 0].T, x[:, 1] - x_old[:, 1].T), axis=0)
            diffs = np.linalg.norm(diffs, axis=0)

            tol = 100
            m = np.amin(diffs, initial=tol)
            while m < tol:
                ind = np.unravel_index(np.argmax(diffs, axis=None), diffs.shape)
                obj.loc[ind(0), ["xold", "yold"]] = x_old[ind(1), 0:2]

                diffs = np.delete(diffs, ind[0], axis=0)
                diffs = np.delete(diffs, ind[1], axis=1)
                m = np.amin(diffs, initial=tol)

            valid_old = ~np.isnan(obj.loc[:, "xold"].values)
            obj.loc[valid_old, ["xnew", "ynew"]] = (
                0.5 * obj.loc[valid_old, ["xold", "xold"]]
                + 0.5 * obj.loc[valid_old, ["xnew", "xnew"]]
            )

            obj["vel"] = (
                np.linalg.norm(
                    obj.loc[:, ["xnew", "ynew"]].values
                    - df.loc[:, ["xold", "xold"]].values,
                    axis=1,
                )
                / 30
            )  # fps

        self.obj_old = obj

        vals = [
            (
                0,
                (i["left"], i["top"]),
                (i["left"] + i["width"], i["top"] + i["height"]),
                i["vel"],
                i["xnew"],
                i["ynew"],
            )
            for i in obj.iterrows()
        ]

        return vals

    def get_frame(self, obj):
        fName = obj.iloc[0, "fileName"][0]
        frame = cv2.imread(fName)
        return frame
