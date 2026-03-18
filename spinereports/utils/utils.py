import numpy as np
import platform
from skimage import measure
from scipy.spatial import KDTree
from spinereports.utils.image import Image, zeros_like
from scipy import interpolate, ndimage
from skimage import measure, morphology

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

        if pos_coords_x.size < 3 or neg_coords_x.size < 3:
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

        seg1 = np.zeros((np.max(np.concatenate([pos_coords_x, neg_coords_x]))+1, np.max(np.concatenate([neg_coords_y, pos_coords_y]))+1))
        seg2 = np.zeros((np.max(np.concatenate([pos_coords_x, neg_coords_x]))+1, np.max(np.concatenate([neg_coords_y, pos_coords_y]))+1))
        if pos_coords.size != 0:
            seg1[pos_coords_x, pos_coords_y] = 1
        if neg_coords.size != 0:
            seg2[neg_coords_x, neg_coords_y] = 1
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

def straighten_coordinates(centerline, spine_centerline, radius):
    # Identify longest canal portion with complete base available
    base_list = []
    for i, (origin, normal) in enumerate(zip(centerline['position'].T, centerline['derivative'].T)):
        superior = normal / np.linalg.norm(normal)  # in the S direction

        # Find anterior vector using spine centerline
        anterior_point = intersect_centerline_plane(spine_centerline, origin, superior)
        if anterior_point is None:
            base_list.append(None)
            continue

        anterior = anterior_point - origin
        anterior = anterior / np.linalg.norm(anterior)  # in the A direction
        left = np.cross(superior, anterior)  # in the L direction

        base_list.append((origin, left, anterior, superior))
    
    # Find longest portion of the centerline with complete base available
    longest_start_idx = None
    longest_length = 0
    current_start_idx = None
    for i, base in enumerate(base_list):
        if base is not None:
            if current_start_idx is None:
                current_start_idx = i
        else:
            if current_start_idx is not None:
                length = i - current_start_idx
                if length > longest_length and current_start_idx <= i-1:
                    longest_length = length
                    longest_start_idx = current_start_idx
                current_start_idx = None
    if current_start_idx is not None:
        length = i - current_start_idx
        if length > longest_length and current_start_idx <= i-1:
            longest_length = length
            longest_start_idx = current_start_idx
    if longest_start_idx is None:
        raise ValueError("Cannot project spine centerline on orhtonal planes of the centerline.")
    
    coordinates = np.zeros((3, 2*radius+1, 2*radius+1, longest_length), dtype=float)
    centerline_start_idx = centerline['position'].T[longest_start_idx,2].astype(int)
    for i, (origin, left, anterior, superior) in enumerate(base_list[longest_start_idx:longest_start_idx+longest_length]):
        steps = np.linspace(start=-radius, stop=+radius, num=2*radius+1)
        coordinates[:, :, :, i] = origin[:, None, None]
        coordinates[:, :, :, i] += steps[None, :, None] * left[:, None, None]
        coordinates[:, :, :, i] += steps[None, None, :] * anterior[:, None, None]
    return coordinates, centerline_start_idx

def intersect_centerline_plane(centerline, point, normal):
    for i in range(len(centerline['position'].T)-1):
        coord_i = centerline['position'].T[i]
        coord_i1 = centerline['position'].T[i+1]
        if np.dot(coord_i - point, normal) * np.dot(coord_i1 - point, normal) <= 0:
            break
        elif i == len(centerline['position'].T)-2:
            return None
    while np.dot(coord_i - coord_i1, normal) > 1e-3:
        mid = (coord_i + coord_i1) / 2
        if np.dot(mid - point, normal) * np.dot(coord_i - point, normal) <= 0:
            coord_i1 = mid
        else:
            coord_i = mid
    intersec_point = (coord_i + coord_i1) / 2
    return intersec_point

def _properties2d(canal, spinalcord, dim, radius=2):
    """
    Compute shape property of the input 2D image. Accounts for partial volume information.
    :param canal: 2D input canal image in uint8 or float (weighted for partial volume) that has a single object.
    :param spinalcord: 2D input spinal cord image in uint8 or float (weighted for partial volume).
    :param dim: [px, py]: Physical dimension of the image (in mm). X,Y respectively correspond to AP,RL.
    :param radius: Radius of the cylindrical mask used for diameter computation.
    :return:
    """
    # Check if slice is empty
    if np.sum(canal) == 0:
        print('The slice is empty.')
        return None

    # Extract canal slice center of mass
    x_pos = canal.shape[0]//2
    y_pos = canal.shape[1]//2
    canal_pos = np.array([x_pos, y_pos])
    
    # Create vector v from canal_pos to spine pos and normalize it
    v = np.array([0, 1])
    v = v / np.linalg.norm(v)

    # Create w an orthogonal vector to v
    w = np.array([1, 0])

    # Compute AP diameter along v 
    v_mask = cylindrical_mask(shape=canal.shape, p0=canal_pos, v=v, radius=radius) # Create cylindrical mask along v
    AP_mask = v_mask*canal
    AP_coords = np.argwhere(AP_mask)
    projections = np.dot(AP_coords, v)  # Project onto vector
    if projections.any():
        diameter_AP_canal = np.mean([np.max(AP_coords[AP_coords[:,0]==row][:,1]) - np.min(AP_coords[AP_coords[:,0]==row][:,1]) for row in np.unique(AP_coords[:, 0])])*dim[0] # AP length = max - min projection
    else:
        diameter_AP_canal = 0
    
    # Compute RL diameter along w
    w_mask = cylindrical_mask(shape=canal.shape, p0=canal_pos, v=w, radius=radius) # Create cylindrical mask along w
    RL_mask = w_mask*canal
    RL_coords = np.argwhere(RL_mask)
    projections = np.dot(RL_coords, w)  # Project onto vector
    if projections.any():
        diameter_RL_canal = np.mean([np.max(RL_coords[RL_coords[:,1]==col][:,0]) - np.min(RL_coords[RL_coords[:,1]==col][:,0]) for col in np.unique(RL_coords[:, 1])])*dim[1] # RL length = max - min projection
    else:
        diameter_RL_canal = 0

    # Compute symmetry score
    asymmetry_canal_R_L = 2 * np.sum(canal * np.flip(canal, axis=0)) / (np.sum(canal) + np.sum(np.flip(canal, axis=0)) + 1e-8)

    # Compute area
    area_canal = np.sum(canal) * dim[0] * dim[1]

    # Compute eccentricity
    if diameter_AP_canal < diameter_RL_canal:
        eccentricity_canal = np.sqrt(1 - diameter_AP_canal**2/diameter_RL_canal**2) if diameter_RL_canal > 0 else 0
    else:
        eccentricity_canal = -np.sqrt(1 - diameter_RL_canal**2/diameter_AP_canal**2) if diameter_AP_canal > 0 else 0

    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity_canal = -1
    else:
        solidity_canal = compute_solidity_2d(canal)

    # Compute spinal cord metrics if not empty else set metrics to -1
    if not np.sum(spinalcord) == 0:
        # Extract spinalcord slice center of mass
        spinalcord_pos = canal_pos.copy()

        # Compute AP diameter along v
        v_mask = cylindrical_mask(shape=spinalcord.shape, p0=spinalcord_pos, v=v, radius=radius) # Create cylindrical mask along v
        AP_mask = v_mask*spinalcord
        AP_coords = np.argwhere(AP_mask)
        projections = np.dot(AP_coords, v)  # Project onto vector
        if projections.any():
            diameter_AP_spinalcord = np.mean([np.max(AP_coords[AP_coords[:,0]==row][:,1]) - np.min(AP_coords[AP_coords[:,0]==row][:,1]) for row in np.unique(AP_coords[:, 0])])*dim[0] # AP length = max - min projection
        else:
            diameter_AP_spinalcord = 0

        # Compute RL diameter along w
        w_mask = cylindrical_mask(shape=spinalcord.shape, p0=spinalcord_pos, v=w, radius=radius) # Create cylindrical mask along w
        RL_mask = w_mask*spinalcord
        RL_coords = np.argwhere(RL_mask)
        projections = np.dot(RL_coords, w)  # Project onto vector
        if projections.any():
            diameter_RL_spinalcord = np.mean([np.max(RL_coords[RL_coords[:,1]==col][:,0]) - np.min(RL_coords[RL_coords[:,1]==col][:,0]) for col in np.unique(RL_coords[:, 1])])*dim[1] # RL length = max - min projection
        else:
            diameter_RL_spinalcord = 0

        # Compute symmetry score
        asymmetry_spinalcord_R_L = 2 * np.sum(spinalcord * np.flip(spinalcord, axis=0)) / (np.sum(spinalcord) + np.sum(np.flip(spinalcord, axis=0)) + 1e-8)

        # Compute area 
        area_spinalcord = np.sum(spinalcord) * dim[0] * dim[1]

        # Compute eccentricity 
        if diameter_AP_spinalcord < diameter_RL_spinalcord:
            eccentricity_spinalcord = np.sqrt(1 - diameter_AP_spinalcord**2/diameter_RL_spinalcord**2) if diameter_RL_spinalcord > 0 else 0
        else:
            eccentricity_spinalcord = -np.sqrt(1 - diameter_RL_spinalcord**2/diameter_AP_spinalcord**2) if diameter_AP_spinalcord > 0 else 0

        # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
        if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
            solidity_spinalcord = -1
        else:
            solidity_spinalcord = compute_solidity_2d(spinalcord)
    else:
        area_spinalcord = -1
        diameter_AP_spinalcord = -1
        diameter_RL_spinalcord = -1
        eccentricity_spinalcord = -1
        solidity_spinalcord = -1
        asymmetry_spinalcord_R_L = -1

    # Fill up dictionary
    properties = {
        'area_canal': area_canal,
        'area_spinalcord': area_spinalcord,
        'diameter_AP_canal': diameter_AP_canal,
        'diameter_AP_spinalcord': diameter_AP_spinalcord,
        'diameter_RL_canal': diameter_RL_canal,
        'diameter_RL_spinalcord': diameter_RL_spinalcord,
        'canal_centroid': canal_pos,
        'eccentricity_canal': eccentricity_canal,
        'eccentricity_spinalcord': eccentricity_spinalcord,
        'solidity_canal': solidity_canal,  # convexity measure
        'solidity_spinalcord': solidity_spinalcord,  # convexity measure
        'asymmetry_canal_R_L': asymmetry_canal_R_L,
        'asymmetry_spinalcord_R_L': asymmetry_spinalcord_R_L
    }
    return properties

def cylindrical_mask(shape, p0, v, radius):
    """
    Create a 2D binary mask of a 'cylinder' (thick line) along vector `v` passing through `p0`.
    
    Args:
        shape (tuple): Shape of the 2D image (height, width)
        p0 (np.array): A point [y, x] the vector passes through
        v (np.array): Direction vector [vy, vx]
        radius (float): Cylinder radius (in pixels)

    Returns:
        mask (2D np.array): Binary mask with True inside the cylinder
    """
    h, w = shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Shift grid by point
    dy = Y - p0[0]
    dx = X - p0[1]
    
    # Normalize direction vector
    v = v / np.linalg.norm(v)
    
    # Compute perpendicular distance to the line (vector projection method)
    # Distance = ||(point - p0) - ((point - p0) · v) * v||
    dot = dx * v[1] + dy * v[0]
    proj_x = dot * v[1]
    proj_y = dot * v[0]
    perp_x = dx - proj_x
    perp_y = dy - proj_y
    dist = np.sqrt(perp_x**2 + perp_y**2)
    
    # Inside mask if distance < radius
    mask = dist < radius
    return mask

def compute_solidity_2d(mask):
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)[0]
    return props.solidity

def project_point_centerline(centerline, ref_coord):
    dist = np.linalg.norm(centerline - ref_coord, axis=1)
    min_idx = np.argmin(dist)
    return centerline[min_idx]    

def get_centerline(seg, smooth=50):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/centerline/core.py

    Extract centerline from canal segmentation using center of mass and interpolate with bspline
    Expect orientation RPI
    '''
    arr = np.array(np.where(seg.data))
    # Loop across SI axis and average coordinates within duplicate SI values
    sorted_avg = []
    for i_si in np.unique(arr[2]):
        sorted_avg.append(arr[:, arr[2] == i_si].mean(axis=1))
    x_mean, y_mean, z_mean = np.array(sorted_avg).T
    z_ref = np.array(range(z_mean.min().astype(int), z_mean.max().astype(int) + 1))

    # Interpolate centerline
    px, py, pz = seg.dim[4:7]
    x_centerline_fit, x_centerline_deriv = bspline(z_mean, x_mean, z_ref, smooth=smooth, pz=pz)
    y_centerline_fit, y_centerline_deriv = bspline(z_mean, y_mean, z_ref, smooth=smooth, pz=pz)

    # Construct output
    arr_ctl = np.array([x_centerline_fit, y_centerline_fit, z_ref])
    arr_ctl_der = np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)])

    # Create centerline dictionary
    centerline = {"position": arr_ctl, "derivative": arr_ctl_der}
    return centerline

def bspline(x, y, xref, smooth, deg_bspline=3, pz=1):
    """
    Copied from https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/centerline/curve_fitting.py
    Bspline interpolation.

    The smoothing factor (s) is calculated based on an empirical formula (made by JCA, based on
    preliminary results) and is a function of pz, density of points and an input smoothing parameter (smooth). The
    formula is adjusted such that the parameter (smooth) produces similar smoothing results than a Hanning window with
    length smooth, as implemented in linear().

    :param x:
    :param y:
    :param xref:
    :param smooth: float: Smoothing factor. 0: no smoothing, 5: moderate smoothing, 50: large smoothing
    :param deg_bspline: int: Degree of spline
    :param pz: float: dimension of pixel along superior-inferior direction (z, assuming RPI orientation)
    :return:
    """
    if len(x) <= deg_bspline:
        deg_bspline -= 2
    density = (float(len(x)) / len(xref)) ** 2
    s = density * smooth * pz / float(3)
    # Then, run bspline interpolation
    tck = interpolate.splrep(x, y, s=s, k=deg_bspline)
    y_fit = interpolate.splev(xref, tck, der=0)
    y_fit_der = interpolate.splev(xref, tck, der=1)
    return y_fit, y_fit_der

def fastest_dilation_edt(mask, radius):
    # Calculate distance from the canal (foreground is 1, background is 0)
    # We want the distance of 0s from the 1s
    distance = ndimage.distance_transform_edt(mask == 0)
    
    # The dilation is where the distance from the original mask is <= radius
    # (Note: we use mask==0, so the distance is 0 inside the canal)
    return distance <= radius