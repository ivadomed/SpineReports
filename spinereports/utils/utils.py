import numpy as np


def find_symmetry_vector_binary(mask, center, angle_step_deg=1.0, refine_window_deg=2.0, refine_step_deg=0.2):
    """
    Find the symmetry axis of a binary 2D mask constrained to pass through a given center.

    Parameters:
        mask: 2D binary array-like
        center: (row, col) tuple for axis anchor
        angle_step_deg: coarse search step in degrees over [0, 180)
        refine_window_deg: local refinement half-window around coarse optimum
        refine_step_deg: local refinement step in degrees

    Returns:
        angle_rad: axis angle in image coordinates (row, col)
        vector_2d: unit direction vector [drow, dcol]
        min_sum: minimum mismatch sum between pos_coords and mirrored neg_coords
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0, np.array([1.0, 0.0], dtype=np.float64), 0.0

    h, w = mask.shape
    center = np.asarray(center, dtype=np.float64)
    coords_f = coords.astype(np.float64)
    rel = coords_f - center[None, :]

    def _sum_for_angle(theta):
        drow = np.cos(theta)
        dcol = np.sin(theta)
        normal = np.array([-dcol, drow], dtype=np.float64)
        sym = np.array([drow, dcol], dtype=np.float64)

        signed_dist = rel @ normal
        pos_coords = coords[signed_dist >= 0]
        neg_coords = coords[signed_dist < 0]

        # Project neg_coords in the sym-normal plane
        neg_coords_x = np.round(np.dot(neg_coords, normal)).astype(int)
        neg_coords_y = np.round(np.dot(neg_coords, sym)).astype(int)

        # Project pos_coords in the sym-normal plane
        pos_coords_x = np.round(np.dot(pos_coords, normal)).astype(int)
        pos_coords_y = np.round(np.dot(pos_coords, sym)).astype(int)

        if pos_coords_x.size == 0 or neg_coords_x.size == 0:
            return np.inf

        # Substract min along x
        min_x = np.min(pos_coords_x)
        pos_coords_x -= min_x
        neg_coords_x -= min_x
        
        # Substract min along y
        min_y = np.min(np.concatenate([neg_coords_y, pos_coords_y]))
        pos_coords_y -= min_y
        neg_coords_y -= min_y

        # Inverse neg_coords_x to mirror them
        neg_coords_x = -neg_coords_x

        seg1 = np.zeros((np.max(np.concatenate([pos_coords_x, neg_coords_x])), np.max(np.concatenate([neg_coords_y, pos_coords_y]))))
        seg2 = np.zeros((np.max(np.concatenate([pos_coords_x, neg_coords_x])), np.max(np.concatenate([neg_coords_y, pos_coords_y]))))
        if pos_coords.size != 0:
            seg1[pos_coords_x-1, pos_coords_y-1] = 1
        if neg_coords.size != 0:
            seg2[neg_coords_x-1, neg_coords_y-1] = 1
        seg = seg1 - seg2
        diff_sum = np.sum(np.abs(seg))
        return diff_sum

    coarse_angles_deg = np.arange(0.0, 180.0, max(angle_step_deg, 0.1))
    coarse_angles = np.deg2rad(coarse_angles_deg)
    coarse_sums = np.array([_sum_for_angle(theta) for theta in coarse_angles], dtype=np.float64)
    best_idx = int(np.argmin(coarse_sums))
    best_angle = float(coarse_angles[best_idx])
    best_sum = float(coarse_sums[best_idx])

    if refine_step_deg > 0 and refine_window_deg > 0:
        local_deg = np.arange(-refine_window_deg, refine_window_deg + refine_step_deg, refine_step_deg)
        local_angles = best_angle + np.deg2rad(local_deg)
        local_sums = np.array([_sum_for_angle(theta) for theta in local_angles], dtype=np.float64)
        local_best = int(np.argmin(local_sums))
        best_angle = float(local_angles[local_best])
        best_sum = float(local_sums[local_best])

    best_angle = float(np.mod(best_angle, np.pi))
    vector_2d = np.array([np.cos(best_angle), np.sin(best_angle)], dtype=np.float64)
    return best_angle, vector_2d, best_sum