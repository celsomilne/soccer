import numpy as np
import cv2
import pandas as pd

from ._base import ModelBase


def merge_similar_lines(l, lines):

	"""
	Merges line l with set lines - a set of unique lines
	Adds l to set if it is unique
	Ignores l if it is contained within another line's neighbourhood
	Extends unique line with l if l is longer
	"""

	# Put unique lines in array if theres only 1, for looping
    if len(lines.shape) == 1:
        lines = np.array([lines])

    # Translate line to align with each unique line
    d = np.column_stack(
        ((lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2)
    )
    d = np.tile(d, (1, 2))
    tl = l - d

    # Rotate line to align with each unique line
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
	
	# Check if line is similar to any unique line
    similar = np.logical_and(abs(tl[:, 1]) < yb, abs(tl[:, 3]) < yb)
    if sum(similar) > 1:
		# If multiple similar lines, take the most similar
        diffs = np.maximum(abs(tl[:, 1]), abs(tl[:, 3]))
        similar[:] = False
        similar[np.argmin(diffs)] = True

    if any(similar):
		# If line is similar, check if it extends beyond current unique line
		# Update unique line to new length if it does
        xb = xb[similar]
        if tl[similar, 0] < -xb or tl[similar, 0] > xb:
            lines[similar, 0:2] = l[0:2]
        if tl[similar, 2] < -xb or tl[similar, 2] > xb:
            lines[similar, 2:4] = l[2:4]
    else:
		# If line is sufficiently differet than other unique lines, add to unique set
        lines = np.concatenate((lines, l.reshape(1, 4)), axis=0)

    return lines


def is_edge_line(l, dims):

	# Tolerance for line to be considered on the edge
    tol = 10
	
	# Check if line is on any 4 borders
    bad = (
        (l[0] < tol and l[2] < tol)
        or (l[1] < tol and l[3] < tol)
        or (l[0] > dims[1] - tol and l[2] > dims[1] - tol)
        or (l[1] > dims[0] - tol and l[3] > dims[0] - tol)
    )
    return bad


def find_intersection(l1, l2):

	# Calculate intercept and gradient of first line
    m1 = (l1[3] - l1[1]) / (l1[2] - l1[0])
    b1 = l1[1] - m1 * l1[0]
	
	# If line is vertical, manually derive intersection
    if l2[0] == l2[2]:
        return np.array([l2[0], m1 * l2[0] + b1])
	
	# Find intercept and gradient of second line
    m2 = (l2[3] - l2[1]) / (l2[2] - l2[0])
    b2 = l2[1] - m2 * l2[0]
	
	# Calculate intercepts of lines
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return np.array([x, y])


def generate_points(line_dict, right_view, true_dict, rel_dict):
	
	# initialise vector/matrix of coefficients
    A = np.array([]).reshape(0, 8)
    b = np.array([])
	
	# Loop through each pair of potential line intercepts to check if they've been detected 
    for n1 in rel_dict.keys():
        if n1 in line_dict:
            for n2 in rel_dict[n1]:
                if n2 in line_dict:
					# Calculate line intercepts on field
                    x = np.array([0.0, 0.0])
                    x += true_dict[n1]
                    if right_view:
                        x += true_dict[n2]
                    else:
                        x -= true_dict[n2]
						
					# Calculate line intercepts in frame
                    u = find_intersection(line_dict[n1], line_dict[n2])
					
					# Add results to coefficient matrix/vector
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
	
	# Add additional 'half' points to straighten far side line if in centre field case
    if right_view == 1:
        u = find_intersection(line_dict["top"], line_dict["left"])
        A = np.vstack((A, [[0, 0, 0, u[0], u[1], 1, -70 * u[0], -70 * u[1]]]))
        u = find_intersection(line_dict["top"], line_dict["right"])
        A = np.vstack((A, [[0, 0, 0, u[0], u[1], 1, -70 * u[0], -70 * u[1]]]))
        b = np.concatenate((b, [70, 70]))
	
	# Calculate homographic transform matrix
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    c = np.concatenate((c, np.array([1.0])))
    c = c.reshape(3, 3)

    return c


class FieldTransform:

	# Initialise the location of the lines on soccer field
    true_dict = {
        'top': np.array([0.,70.]).astype(np.float32),
        'mid': np.array([0.,0.]).astype(np.float32),
        'box': np.array([33.5,0.]).astype(np.float32),
        'tbox': np.array([0.,55.15]).astype(np.float32),
        'bbox': np.array([0.,14.85]).astype(np.float32),
        'goal': np.array([50.,0.]).astype(np.float32),
        'tcirc': np.array([0.,44.15]).astype(np.float32),
        'bcirc': np.array([0.,25.85]).astype(np.float32),
        'rcirc': np.array([9.15,0.]).astype(np.float32),
        'lcirc': np.array([-9.15,0.]).astype(np.float32),
        'ccirc': np.array([0.,35.]).astype(np.float32)
    }
	
	# Initialise dictionary of line intercepts
    rel_dict = {
        'top': ['mid','box','goal'],
        'tbox': ['box','goal'],
        'bbox': ['box','goal'],
        'tcirc': ['mid'],
        'bcirc': ['mid'],
        'ccirc': ['lcirc','rcirc'],
    }

    def __call__(self, im):
	
		# Creake a mask to extract the field
        mask_field = np.logical_and(im[:,:,1] > im[:,:,0], im[:,:,1] > im[:,:,2]).astype(np.uint8)
		
		# Use morphological operations to clear noise from th mask
        k = 40
        kernel = np.ones((k,k),np.uint8)
        mask = cv2.morphologyEx(mask_field, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2*k,2*k),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		
		# Apply mask back on to image to get field
        field = cv2.bitwise_and(im, im, mask = mask)
		
		# Mildly decay mask to remove edges generated by mask edge
        kernel = np.ones((1,1),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
		
		# Use blue channel to extract edged as maximally exposes soccer lines
        gray = im[:,:,2]
		
		# Apply Canny edge detection technique
        blur_gray = cv2.GaussianBlur(gray,(5, 5),0)
        low_threshold = 30
        high_threshold = 200
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges = cv2.bitwise_and(edges, edges, mask = mask)
        edges = cv2.dilate(edges, np.ones((3,3)))
		
		# Find lines in image using probabilistic Hough transform
        rho_tol = 1
        theta_tol = np.pi/180
        lines = np.array(cv2.HoughLinesP(edges,rho_tol,theta_tol,50,minLineLength=100,maxLineGap=5))
        lines.shape = (lines.shape[0], lines.shape[2])
		
		# Remove lines on border and merge remaining multiple detections
        good_lines = np.array([l for l in lines if not is_edge_line(l,edges.shape)])
        lines = good_lines[0]
        for i in range(1,len(good_lines)):
            lines = merge_similar_lines(good_lines[i],lines)
		
		# Direction of each line
        theta = np.arctan2(lines[:,3]-lines[:,1], lines[:,2]-lines[:,0])%np.pi
		
		# Initialise dictionary to store classified lines
        line_dict = {'left': np.array([0,0,0,edges.shape[0]]),
                    'right': np.array([edges.shape[1],0,edges.shape[1],edges.shape[0]])}
        right_view = 2
		
		# Check if near vertical line in image (centre field indicator)
        if np.any(abs(theta - np.pi/2) < np.pi/60)  or len(theta) < 2:
			# Update view flag for centre field
            right_view = 1
			
			#Classify halfway line and far sideline as most and horizontal lines respectively 
            line_dict['mid'] = lines[np.argmin(abs(theta-np.pi/2))]
            line_dict['top'] = lines[np.pi/2 - abs(theta - np.pi/2) < np.pi/30][0]
            midpoint = (line_dict['mid'][0] + line_dict['mid'][2])//2
			
			# Remove lines and boarder from the edge output
            line_mask = np.ones(gray.shape, np.uint8)
            for x1,y1,x2,y2 in line_dict.values():
                cv2.line(line_mask,(x1,y1), (x2,y2), (0,0,0),10)
            line_mask[:line_mask.shape[0]//8,:] = 0
            line_mask[(line_mask.shape[0] - line_mask.shape[0]//8):line_mask.shape[0],:] = 0
            line_mask[:,:20] = 0
            line_mask[:,line_mask.shape[1]-20:] = 0
            edges = cv2.bitwise_and(edges, edges, mask = line_mask)
            
			# Find the largest remaining contour in image (centre circle)
            conts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(conts, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            
			# Find closest remaining lines to the halfway line
			# (these lines form the edges of the ellipse)
            line_disp = lines[:,[0,3]] - midpoint
            idx = np.argpartition(np.mean(line_disp,axis=1) + np.min(line_disp,axis=1), 2)
            lines = lines[np.any(lines != line_dict['top'],axis=1)]
            lines = lines[np.any(lines != line_dict['mid'],axis=1)]
            lines = lines[idx[0:2]]
			
			# Classify these lines as centre circle intercept with the halfway line
            line_dict['tcirc'] = lines[np.argmin(np.mean(lines[:,[1,3]],axis=1))]
            line_dict['bcirc'] = lines[np.argmax(np.mean(lines[:,[1,3]],axis=1))]
			
			# Find halfway point between this intersection
            c_top = find_intersection(line_dict['tcirc'], line_dict['mid'])
            c_bot = find_intersection(line_dict['bcirc'], line_dict['mid'])
            c_cen = (c_top[1] + c_bot[1])/2
            line_dict['ccirc'] = np.array([x,c_cen,x+w,c_cen])
			
			# From largest contour, determine whether it lies to left or right of halfway line
			# Classify an extreme point of centre circle accordingly
            if x < midpoint:
                line_dict['lcirc'] = np.array([x,y,x,y+h])
            else:
                line_dict['rcirc'] = np.array([x+w,y,x+w,y+h])
        
		# No halfway line (end field indicator)
		else:
		
			# Classify horizontal lines on field as minimum line orientation
            horz_theta = theta[np.argmin(np.pi/2-abs(theta-np.pi/2))]
			
			# Determine if at left or right end of field based on horizontal orientation
            if horz_theta < np.pi/2:
                right_view = 0
            
			#Extract all horizontal lines in image
            horz_lines = lines[abs(theta-horz_theta) < np.pi/30]
			
			# Classify far sideline as heightest horizontal line and remove from sets
            line_dict['top'] = horz_lines[np.argmin(np.minimum(horz_lines[:,1], horz_lines[:,3]))]
            lines = lines[np.any(lines != line_dict['top'],axis=1)]
            horz_lines = horz_lines[np.any(horz_lines != line_dict['top'],axis=1)]
			
			# Check if goal line in image by comparing endpoints with far sideline
            if right_view:
                goal_line_check = np.logical_and(lines[:,0]-line_dict['top'][2]<100,\
                                            lines[:,1]-line_dict['top'][3]<20)
            else:
                goal_line_check = np.logical_and(lines[:,2]-line_dict['top'][0]<100,\
                                            lines[:,3]-line_dict['top'][1]<20)
			
			# Classify any positive hits as the goal line
            if any(goal_line_check):
                line_dict['goal'] = lines[goal_line_check][0]
                lines = lines[np.logical_not(goal_line_check)]
			
			# Classify far pentaly box line as next heighest horizontal line and remove
            line_dict['tbox'] = horz_lines[np.argmin(horz_lines[:,1]+horz_lines[:,3])]
            lines = lines[np.any(lines != line_dict['tbox'],axis=1)]
            horz_lines = horz_lines[np.any(horz_lines != line_dict['tbox'],axis=1)]

			# Check if any horizontal lines remain and classify next heightest as close penalty box line
            if len(horz_lines) != 0 and 'tbox' in line_dict:
                bbox_temp = horz_lines[np.argmin(horz_lines[:,1]+horz_lines[:,3])]
                if np.max(bbox_temp[[1,3]]) - np.max(line_dict['tbox'][[1,3]]) > \
                        2*(np.max(line_dict['tbox'][[1,3]]) - np.max(line_dict['top'][[1,3]])):
                    line_dict['bbox'] = bbox_temp            
                    lines = lines[np.any(lines != line_dict['bbox'],axis=1)]
                    horz_lines = horz_lines[np.any(horz_lines != line_dict['bbox'],axis=1)]
					
			# Check if vertical pentaly box line in image by comparing endpoints
            if right_view:
                box_line_check = np.linalg.norm(lines[:,0:2]-line_dict['tbox'][0:2],axis=1) < 50
            else:
                box_line_check = np.linalg.norm(lines[:,2:4]-line_dict['tbox'][2:4],axis=1) < 50
				
			# Classify any positive hits as the vertical penalty box line
            if any(box_line_check):
                line_dict['box'] = lines[box_line_check][0]
                lines = lines[np.logical_not(box_line_check)]
		
		# Generate transform using the classified lines from the image
        c = generate_points(line_dict, right_view, self.true_dict, self.rel_dict)

        return c

class SoccerModel:
    ft = FieldTransform()
    obj_old = pd.DataFrame()
    c = np.zeros((3, 3))
    count = 6

    def __init__(self, detector):
        self.detector = detector
        self.objects = detector.bb_df

    def predict(self, frameIdx, fps=25):

        # Pull objects in current dataframe
        df = self.detector.bb_df
        obj = self.detector.get_df(batchNum=frameIdx).reset_index()
        frame = self.get_frame(obj)
        if frame is None:
            return None

        width = obj["width"]
        height = obj["height"]

        # Get only rows for which the width and height are within 2 standard deviations
		# (Extracts players)
        obj = obj[np.abs(width - np.mean(width)) / np.std(width) < 2]
        obj = obj[np.abs(height - np.mean(height)) / np.std(height) < 2]

        # Get the locations of the feet
        obj["ufeet"] = obj["left"] + obj["width"] / 2
        obj["vfeet"] = obj["top"] + obj["height"]

        # Get the right and bottom bounding boxes
        obj["right"] = obj["left"] + obj["width"]
        obj["bottom"] = obj["top"] + obj["height"]

        # Get the feature transform if available and not corrupt
        try:
            c = self.ft(frame)
            if (np.abs(c[0,2]) > .01 and np.abs(c[1,2]) > .01 and self.count > 5) or frameIdx == 1:
                self.c = c
                self.count = 0
        except Exception as e:
            pass
        finally:
            self.count += 1
		
		# Stack feet coordinates to be transformed
        u = np.vstack(
            (obj["ufeet"].values, obj["vfeet"].values, np.ones((1, len(obj))))
        )

        # Homographic transform
        # u  - frame coordinates
        # xy - field coordinates
        xy = self.c.dot(u).T
        xy = np.divide(xy.T, xy[:, 2]).T

        # Initialise the new fieled positions as 0
        obj["xnew"] = 0
        obj["ynew"] = 0

        # Set the field positions equal to the transformed coordinates
        obj[["xnew", "ynew"]] = xy[:, 0:2]

        if frameIdx == 1:

            # Store data and abort
            obj["vel"] = 0.0
            self.obj_old = obj
        else:

            # Extract the old x and y coordinates
            xy_old = self.obj_old.loc[:, ["xnew", "ynew"]].values.copy()

            # Take the euclidean difference of the old and new positions
            diffs = np.stack(
                (
                    xy[:, 0]
                    - xy_old[:, 0].reshape((-1, 1)),  # difference in x position
                    xy[:, 1]
                    - xy_old[:, 1].reshape((-1, 1)),  # difference in y position
                ),
                axis=0,
            )
            diffs = np.linalg.norm(diffs, axis=0)

            # Initialise 'old' positions
            obj["xold"] = 0
            obj["yold"] = 0

            # Maximum euclidean distance between matching players
            tol = 50

            # Iterate until tolerance met or no objects in frame
            while np.any(diffs < tol) and len(obj) > 0 and len(self.obj_old) > 0:
                ind = np.unravel_index(np.argmin(diffs, axis=None), diffs.shape)
                obj.loc[ind[1], ["xold", "yold"]] = xy_old[ind[0], 0:2]
                diffs[:, ind[1]] = np.inf
                diffs[ind[0], :] = np.inf

            # Filter the x and y positions
            valid_pos = ~np.isnan(obj['xold'])
            obj.loc[valid_pos, ["xnew", "ynew"]] = 0.05 * obj.loc[valid_pos, ["xnew", "ynew"]].values + \
                0.95 * obj.loc[valid_pos, ["xold", "yold"]].values

            # Find the speed of the players
            norm = np.linalg.norm(
                obj.loc[:, ["xnew", "ynew"]].values
                - obj.loc[:, ["xold", "yold"]].values,
                axis=1,
            )
            obj["vel"] = norm * fps

            # Cap speed at 10
            obj.loc[obj['vel'] > 10, 'vel'] = 10
		
		# Store new objects for next frame
        self.obj_old = obj
		
		# Format data for display functions
        vals = [
            (
                i["label"],
                (
                    (i["left"], i["top"]),
                    (i["left"] + i["width"], i["top"] + i["height"]),
                ),
                i["vel"],
                (i["ynew"] / 70),
                (i["xnew"] + 50) / 100,
            )
            for idx, i in obj.iterrows()
            if (i["left"] == i["left"])
        ]

        return vals

    def get_frame(self, obj):
        try:
            fName = obj.loc[0, "fileName"][0]
            frame = cv2.imread(fName)
            return frame
        except Exception as e:
            return None
