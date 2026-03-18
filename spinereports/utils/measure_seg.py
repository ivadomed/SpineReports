import json, os, textwrap, argparse
import multiprocessing as mp
import importlib
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import numpy as np
from skimage import measure, morphology
from scipy import ndimage as ndi
import platform
import csv
import warnings
import cv2
from sklearn.mixture import GaussianMixture
import colorsys

from spinereports.utils.image import Image, resample_nib, zeros_like
from spinereports.utils.utils import find_symmetry_vector_binary, straighten_coordinates, _properties2d, project_point_centerline, get_centerline, fastest_dilation_edt
from skimage.morphology import ball, binary_dilation
import totalspineseg.resources as resources

import SimpleITK as sitk
from scipy.spatial import ConvexHull
from scipy import ndimage

warnings.filterwarnings("ignore")


def _build_fixed_label_lut(max_label: int) -> np.ndarray:
    """Return a deterministic RGB lookup-table for label values [0..max_label]."""
    if max_label < 0:
        max_label = 0
    lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    lut[0] = (0, 0, 0)
    for label_value in range(1, max_label + 1):
        # Golden-ratio spacing for stable, well-separated hues.
        hue = (label_value * 0.61803398875) % 1.0
        sat = 0.75
        val = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        lut[label_value] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def n4_bias_field_correction(image_data, mask, shrink_factor=2, number_of_histogram_bins=200, 
                           maximum_number_of_iterations=[50, 50, 30, 20], 
                           convergence_threshold=1e-6, spline_order=3):
    """
    Apply N4 bias field correction to MRI image data using SimpleITK.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        3D MRI image data to be corrected
    mask : numpy.ndarray
        Binary mask defining the region for bias correction
    shrink_factor : int, default=2
        Factor to shrink the image for faster processing (1 = no shrinking, 2 = 2x faster, 4 = 4x faster)
    number_of_histogram_bins : int, default=200
        Number of histogram bins for bias field estimation
    maximum_number_of_iterations : list, default=[50, 50, 50, 50]
        Maximum iterations for each resolution level
    convergence_threshold : float, default=1e-6
        Convergence threshold for the algorithm
    spline_order : int, default=3
        B-spline order for bias field modeling
        
    Returns:
    --------
    numpy.ndarray
        Bias-corrected image data
    """
    
    try:
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image_data.astype(np.float32))
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        # Set up N4 bias correction filter
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(maximum_number_of_iterations)
        corrector.SetConvergenceThreshold(convergence_threshold)
        corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
        corrector.SetWienerFilterNoise(0.01)
        corrector.SetNumberOfHistogramBins(number_of_histogram_bins)
        corrector.SetSplineOrder(spline_order)

        # Apply bias correction with optional shrinking for speed
        if shrink_factor > 1:
            # Shrink image and mask for faster processing
            shrink_factors = [shrink_factor] * sitk_image.GetDimension()
            sitk_image_shrunk = sitk.Shrink(sitk_image, shrink_factors)
            sitk_mask_shrunk = sitk.Shrink(sitk_mask, shrink_factors)
            
            # Run N4 on shrunk images to get bias field
            corrector.Execute(sitk_image_shrunk, sitk_mask_shrunk)
            
            # Get the log bias field from the shrunk processing
            log_bias_field_shrunk = corrector.GetLogBiasFieldAsImage(sitk_image_shrunk)
            
            # Resample the bias field back to original resolution
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitk_image)
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetDefaultPixelValue(0.0)
            log_bias_field = resampler.Execute(log_bias_field_shrunk)
            
            # Apply bias correction to original image
            bias_field = sitk.Exp(log_bias_field)
            corrected_image = sitk.Divide(sitk_image, bias_field)
        else:
            # Apply bias correction directly without shrinking
            corrected_image = corrector.Execute(sitk_image, sitk_mask)
        
        # Convert back to numpy array
        corrected_data = sitk.GetArrayFromImage(corrected_image)

        return corrected_data.astype(image_data.dtype)

    except Exception as e:
        print(f"Warning: N4 bias correction failed with error: {e}")
        print("Returning original image without bias correction.")
        return image_data


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) image and segmentation files.
            It uses MRI scans and totalspineseg segmentations to extract metrics from the canal, the discs and vertebrae.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_measure_seg -i images -s segmentations -o metrics
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--segs-dir', '-s', type=Path, required=True,
        help='The folder where input NIfTI segmentation files are located (required).'
    )
    parser.add_argument(
        '--labels-dir', '-l', type=Path, required=True,
        help='The folder where input NIfTI labels (at the posterior tip of the discs) files are located (required).'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where output metrics will be saved (required).'
    )
    parser.add_argument(
        '--prefix', '-p', type=str, default='',
        help='File prefix to work on.'
    )
    parser.add_argument(
        '--image-suffix', type=str, default='_0000',
        help='Image suffix, defaults to "_0000".'
    )
    parser.add_argument(
        '--seg-suffix', type=str, default='',
        help='Segmentation suffix, defaults to "".'
    )
    parser.add_argument(
        '--label-suffix', type=str, default='',
        help='Label suffix, defaults to "".'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    images_path = args.images_dir
    segs_path = args.segs_dir
    labels_path = args.labels_dir
    ofolder = args.ofolder
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    label_suffix = args.label_suffix
    max_workers = args.max_workers
    quiet = args.quiet

    # Use default mapping path
    resources_path = importlib.resources.files(resources)
    mapping_path = os.path.join(resources_path, 'labels_maps/tss_map.json')

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            segs_path = "{segs_path}"
            labels_path = "{labels_path}"
            ofolder = "{ofolder}"
            prefix = "{prefix}"
            image_suffix = "{image_suffix}"
            seg_suffix = "{seg_suffix}"
            label_suffix = "{label_suffix}"
            mapping_path = "{mapping_path}"
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    measure_seg_mp(
        images_path=images_path,
        segs_path=segs_path,
        labels_path=labels_path,
        ofolder_path=ofolder,
        prefix=prefix,
        image_suffix=image_suffix,
        seg_suffix=seg_suffix,
        label_suffix=label_suffix,
        mapping_path=mapping_path,
        max_workers=max_workers,
        quiet=quiet,
    )

def measure_seg_mp(
        images_path,
        segs_path,
        labels_path,
        ofolder_path,
        prefix='',
        image_suffix='_0000',
        seg_suffix='',
        label_suffix='',
        mapping_path='',
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    segs_path = Path(segs_path)
    labels_path = Path(labels_path)
    ofolder_path = Path(ofolder_path)

    glob_pattern = f'{prefix}*{image_suffix}.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    seg_path_list = [segs_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{seg_suffix}.nii.gz') for image_path in image_path_list]
    labels_path_list = [labels_path / image_path.relative_to(images_path).parent / image_path.name.replace(f'{image_suffix}.nii.gz', f'{label_suffix}.nii.gz') for image_path in image_path_list]

    # Load mapping
    with open(mapping_path, 'r') as file:
        mapping = json.load(file)

    process_map(
        partial(
            _measure_seg,
            ofolder_path=ofolder_path,
            mapping=mapping,
        ),
        image_path_list,
        seg_path_list,
        labels_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _measure_seg(
        img_path,
        seg_path,
        label_path,
        ofolder_path,
        mapping
    ):
    '''
    Wrapper function to handle IO.
    '''
    # Load image and segmentation
    img = Image(str(img_path)).change_orientation('RPI')
    seg = Image(str(seg_path)).change_orientation('RPI')
    label = Image(str(label_path)).change_orientation('RPI')

    metrics = {}
    imgs = {}
    try:
        metrics, imgs = measure_seg(
            img=img,
            seg=seg,
            label=label,
            mapping=mapping,
        )
    except ValueError as e:
        print(f'ValueError: {seg_path}, {e}')
        return
    except KeyError as e:
        print(f'KeyError: {seg_path}, {e}')
        return
    except IndexError as e:
        print(f'IndexError: {seg_path}, {e}')
        return
    except Exception as e:
        print(f'Error: {seg_path}, {e}')
        return
    
    # Create output folders if does not exists
    img_name=Path(str(seg_path)).name.replace('.nii.gz', '')
    ofolder_path = Path(os.path.join(ofolder_path, img_name))
    csv_folder_path = ofolder_path / 'csv'
    imgs_folder_path = ofolder_path / 'imgs'
    csv_folder_path.mkdir(parents=True, exist_ok=True)
    imgs_folder_path.mkdir(parents=True, exist_ok=True)

    # Save sagittal image and segmentation
    sagittal_slice = img.change_orientation('RSP').data[img.data.shape[0] // 2, :, :]
    sagittal_slice_p5 = np.percentile(sagittal_slice, 5)
    sagittal_slice_p95 = np.percentile(sagittal_slice, 95)
    sagittal_slice = (sagittal_slice - sagittal_slice_p5) / (sagittal_slice_p95 - sagittal_slice_p5 + 1e-8)
    sagittal_u8 = np.clip(sagittal_slice * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(imgs_folder_path / 'img.png'), sagittal_u8)

    # Also save a sagittal image with segmentation overlay using a fixed colormap.
    seg_sag = seg.change_orientation('RSP').data[seg.data.shape[0] // 2, :, :]
    seg_sag = np.nan_to_num(seg_sag, nan=0).astype(np.int32)
    max_label = 0
    try:
        max_label = int(max(mapping.values()))
    except Exception:
        max_label = 0
    max_label = int(max(max_label, int(seg_sag.max()) if seg_sag.size else 0))
    lut_rgb = _build_fixed_label_lut(max_label)
    seg_rgb = lut_rgb[np.clip(seg_sag, 0, max_label)]

    base_rgb = cv2.cvtColor(sagittal_u8, cv2.COLOR_GRAY2RGB)
    overlay_rgb = base_rgb.copy()
    mask = seg_sag > 0
    alpha = 0.75
    if np.any(mask):
        overlay_rgb[mask] = (
            (1.0 - alpha) * base_rgb[mask].astype(np.float32)
            + alpha * seg_rgb[mask].astype(np.float32)
        ).astype(np.uint8)

    # OpenCV expects BGR
    overlay_bgr = overlay_rgb[..., ::-1]
    cv2.imwrite(str(imgs_folder_path / 'seg_overlay.png'), overlay_bgr)
    cv2.imwrite(str(imgs_folder_path / 'raw_and_seg_overlay.png'), np.concatenate((base_rgb, overlay_bgr), axis=1))

    # Save csv files
    for struc in metrics.keys():
        if len(metrics[struc]) != 0:
            csv_name = f'{struc}.csv'
            csv_path = csv_folder_path / csv_name
            fieldnames=list(metrics[struc][0].keys())
            with open(str(csv_path), mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics[struc]:
                    writer.writerow(row)
    
    # Save images
    for name, img in imgs.items():
        img_name = f'{name}.png'
        img_path = imgs_folder_path / img_name
        # Scale image to [0, 255] and convert to uint8
        if 'foramen' in img_name:
            img_to_save = np.clip(img * 125, 0, 255).astype(np.uint8)
        else:
            img_to_save = np.clip(img * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(img_path), img_to_save)
    
def measure_seg(img, seg, label, mapping):
    '''
    Compute morphometric measurements of the spinal canal, the intervertebral discs and the neural foramen
    '''
    # Fetch unique segmentation values
    unique_seg = np.unique(seg.data)

    # Resample image to isotropic resolution
    nx, ny, nz, nt, px, py, pz, pt = seg.dim
    pr = min([px, py, pz])
    seg = resample_nib(seg, new_size=[pr, pr, pr], new_size_type='mm', interpolation='nn', verbose=False)
    img = resample_nib(img, new_size=[pr, pr, pr], new_size_type='mm', interpolation='linear', verbose=False)

    # Apply N4 algorithm to correct for intensity non uniformity
    img.data = n4_bias_field_correction(img.data, seg.data > 0)

    # Normalize image intensity
    img.data = (img.data - np.mean(img.data)) / np.std(img.data) # Normalize with mean and std

    # Extract spinal canal from segmentation (CSF + SC)
    seg_canal = zeros_like(seg)
    seg_canal.data[seg.data == mapping['SC']] = 1
    seg_canal.data[seg.data == mapping['CSF']] = 2

    # Extract canal centerline
    centerline = get_centerline(seg_canal)

    # Compute centerline distance
    distance = np.array([0] + [np.linalg.norm(centerline['position'][:,i] - centerline['position'][:,i-1]) for i in range(1, centerline['position'].shape[1])]).cumsum()
    centerline_distance = {int(centerline['position'][2, i]): distance[i] for i in range(centerline['position'].shape[1])}

    # Project discs label coordinates on canal centerline
    discs_label = np.array(label.getNonZeroCoordinates(sorting='z'))
    proj_discs_label = [np.concatenate([np.round(project_point_centerline(centerline['position'].T, disc_label[:3])).astype(int), np.array([disc_label[-1]])]) for disc_label in discs_label]

    # Create dict with z-slice and values for discs posterior tip
    disc_slices = {}
    for x, y, z, v in proj_discs_label:
        # Rescale z base on image resolution
        z_rescaled = int(round(z * pz / pr))
        disc_slices[z_rescaled] = v

    # Init output dictionaries with metrics
    metrics = {}
    imgs = {}

    # Measure CSF signal
    seg_csf_data = (seg.data == mapping['CSF']).astype(int)
    properties = measure_csf(img.data, seg_csf_data)
    csf_signal = properties['csf_signal']

    p5 = np.percentile(img.data, 5)
    p95 = np.percentile(img.data, 95)
    img.data = (img.data - p5) / (csf_signal - p5 + 1e-8)
    img.data = np.clip(img.data, 0, 1)
    csf_signal = 1.0

    rows = []
    for i, (k, v) in enumerate(properties['slice_signal'].items()):
        row = {
            "structure": "csf",
            "index": i,
            "slice_nb": k,
            "disc_level": disc_slices[k] if k in disc_slices else None,
            "centerline_distance": centerline_distance[k],
            "slice_signal": v
            }

        rows.append(row)
    metrics['csf'] = rows
    
    # Compute metrics onto vertebrae
    vertebrae_rows = []
    body_dict = {}
    seg_bin = zeros_like(seg) # Segmentation of the vertebral bodies and the discs
    for i, struc in enumerate(mapping.keys()):
        if mapping[struc] in unique_seg and (10 < mapping[struc] < 50): # Vertebrae
            if not struc in body_dict.keys():
                # Compute vertebrae properties
                seg_vert_data = (seg.data == mapping[struc]).astype(int)
                # Check if vertebra is more than one slice
                if (seg_vert_data.sum(axis=0).sum(axis=0)).astype(bool).sum() > 1:
                    properties, img_dict, body_array, add_struc = measure_vertebra(img_data=img.data, seg_vert_data=seg_vert_data, seg_canal_data=seg_canal.data, canal_centerline=centerline, pr=pr)
                    
                    if add_struc:
                        # Save image
                        imgs[f'vertebrae_{struc}_seg'] = img_dict['seg']
                        imgs[f'vertebrae_{struc}_img'] = img_dict['img']

                        # Add vertebral bodies
                        seg_bin.data[body_array.astype(bool)] = 1

                        # Create a row per position/thickness point
                        vertebrae_row = {
                            "structure": "vertebra",
                            "name": struc,
                            "AP_thickness": properties['AP_thickness'],
                            "median_thickness": properties['median_thickness'],
                            "center": properties['center'],
                            "volume": properties['volume'],
                            "median_signal": properties['median_signal'],
                            "ap_attenuation": properties['ap_attenuation']
                        }
                        vertebrae_rows.append(vertebrae_row)
                        body_dict[struc] = body_array
            
    metrics['vertebrae'] = vertebrae_rows

    # Extract signal homogeneity from vertebrae signal
    vert_signal = [vert_dict['median_signal'] for vert_dict in vertebrae_rows]
    vert_signal_dict = {vert_dict["name"]: (vert_dict['median_signal']*vert_dict['ap_attenuation'])/np.max(vert_signal) for vert_dict in vertebrae_rows}
    
    # Add intervertebral discs to seg_bin
    for struc in mapping.keys():
        if mapping[struc] in unique_seg and '-' in struc: # Intervertbral disc in segmentation
            seg_disc_data = (seg.data == mapping[struc]).astype(int)
            seg_bin.data[seg_disc_data.astype(bool)] = 1

    # Create spine centerline using vertebral bodies and discs
    if 50 in unique_seg: # Add sacrum
        seg_bin.data[seg.data == 50] = 1
    dilation_radius = 1.5//pr  # 1.5 mm
    seg_bin.data = binary_dilation(seg_bin.data, ball(dilation_radius)) # Dilate seg_bin to remove gap between discs and vertebrae
    spine_centerline = get_centerline(seg_bin, smooth=500) # Smooth more than canal centerline to get a smoother spine centerline

    # # Show spine centerline and canal centerline
    # for i in range(3):
    #     canal_slice = np.argmax(seg_canal.data, axis=i)
    #     spine_slice = np.argmax(seg_bin.data, axis=i)*2
    #     img_slice = canal_slice + spine_slice
    #     img_slice_rgb = cv2.cvtColor(
    #         img_slice.astype(np.uint8),
    #         cv2.COLOR_GRAY2RGB
    #     )
    #     idx = [j for j in range(3) if j != i]
    #     for coords in np.round(centerline['position']).astype(int).T:
    #         img_slice_rgb[coords[idx[0]], coords[idx[1]]] = [0, 0, 1] # Red overlay
    #     for coords in np.round(spine_centerline['position']).astype(int).T:
    #         img_slice_rgb[coords[idx[0]], coords[idx[1]]] = [0, 1, 0] # Green overlay
    #     os.makedirs(f'test/canal/', exist_ok=True)
    #     cv2.imwrite(f'test/canal/projection_{i}.png', img_slice_rgb*125)

    # Compute metrics onto canal segmentation
    properties = measure_canal(seg_canal, centerline, spine_centerline)
    rows = []
    for i in range(len(properties[list(properties.keys())[0]])):
        slice_nb = list(properties[list(properties.keys())[0]].keys())[i]
        row = {
            "structure": "canal",
            "index": i,
            "slice_nb": slice_nb,
            "disc_level": disc_slices[slice_nb] if slice_nb in disc_slices else None,
            "centerline_distance": centerline_distance[slice_nb],
            }
        for key in properties.keys():
            row[key] = properties[key][slice_nb]
        rows.append(row)
    metrics['canal'] = rows

    # Compute metrics onto foramens
    foramens_rows = []
    for i, struc in enumerate(body_dict.keys()):
        vert_value = int(struc[1:])
        if struc.startswith('C'):
            if vert_value == 7:
                next_vert = 'T1'
            else:
                next_vert = f'C{vert_value+1}'
        elif struc.startswith('T'):
            if vert_value == 12:
                next_vert = 'L1'
            else:
                next_vert = f'T{vert_value+1}'
        elif struc.startswith('L'):
            next_vert = f'L{vert_value+1}'
        if next_vert in body_dict.keys(): # two adjacent vertebrae
            # Fetch vertebrae names
            top_vert = struc
            bottom_vert = next_vert
            foramens_name = f'foramens_{top_vert}-{bottom_vert}'

            # Init foramen segmentation
            if f'{top_vert}-{bottom_vert}' != 'C1-C2':
                disc_mask = (seg.data == mapping[f'{top_vert}-{bottom_vert}'])
                seg_foramen_data = disc_mask.astype(int) * 2 # Set disc value to 2
            else:
                seg_foramen_data = np.zeros_like(seg.data).astype(int)

            # Compute vertebrae properties
            for vert in [top_vert, bottom_vert]:
                seg_foramen_data += (seg.data == mapping[vert]).astype(int)
            
        elif struc.startswith('L') and vert_value > 4 and mapping['sacrum'] in unique_seg:
            # Create foramen name
            foramens_name = f'foramens_{struc}-S'

            # Init foramen segmentation
            disc_mask = (seg.data == mapping[f'L5-S'])
            seg_foramen_data = disc_mask.astype(int) * 2 # Set disc value to 2

            # Add vertebrae and sacrum
            seg_foramen_data += (seg.data == mapping[struc]).astype(int)
            seg_foramen_data += (seg.data == mapping["sacrum"]).astype(int) * 3  # Set sacrum value to 3
        
        else:
            continue

        # Compute foramens properties
        foramens_areas, foramens_img, foramens_seg = measure_foramens(foramens_name=foramens_name, img_data=img.data, seg_foramen_data=seg_foramen_data, seg_canal_data=seg_canal.data, canal_centerline=centerline, spine_centerline=spine_centerline, pr=pr)
    
        # Save images
        for side,im in foramens_seg.items():
            imgs[f'{foramens_name}_{side}_seg'] = im
        
        for side,im in foramens_img.items():
            imgs[f'{foramens_name}_{side}_img'] = im
        
        # Save foramen metrics
        foramens_row = {
            "structure": "foramen",
            "name": foramens_name,
            "right_surface": foramens_areas['right'],
            "left_surface": foramens_areas['left']
        }
        foramens_rows.append(foramens_row)

    # Compute foramen compression ratio using all extracted foramens
    metrics['foramens'], imgs = compute_foramen_compression_ratio(foramens_rows, imgs)

    # Compute metrics onto intervertebral discs
    rows = []
    for struc in mapping.keys():
        if mapping[struc] in unique_seg and '-' in struc: # Intervertbral disc in segmentation
            seg_disc_data = (seg.data == mapping[struc]).astype(int)
            # Check if disc is more than one slice
            if (seg_disc_data.sum(axis=0).sum(axis=0)).astype(bool).sum() > 1:
                lower_vert = struc.split('-')[0]
                upper_vert = struc.split('-')[1]
                # Apply intensity coeff based on vertebrae signal homogeneity to account for signal loss
                if lower_vert in vert_signal_dict and upper_vert in vert_signal_dict:
                    intensity_coeff = (vert_signal_dict[lower_vert] + vert_signal_dict[upper_vert]) / 2
                elif upper_vert in vert_signal_dict:
                    intensity_coeff = vert_signal_dict[upper_vert]
                else:
                    intensity_coeff = 1
                properties, img_dict, add_struc = measure_disc(img_data=img.data, seg_disc_data=seg_disc_data, centerline=centerline, csf_signal=csf_signal*intensity_coeff, pr=pr)

                if add_struc:
                    # Save image
                    imgs[f'discs_{struc}_seg'] = img_dict['seg']
                    imgs[f'discs_{struc}_img'] = img_dict['img']
                    # Create a row
                    row = {
                        "structure": "disc",
                        "name": struc,
                        "eccentricity_AP-RL": properties['eccentricity_AP-RL'],
                        "eccentricity_AP-SI": properties['eccentricity_AP-SI'],
                        "eccentricity_RL-SI": properties['eccentricity_RL-SI'],
                        "solidity": properties['solidity'],
                        "nucleus_eccentricity_AP-RL": properties['nucleus_eccentricity_AP-RL'],
                        "nucleus_eccentricity_AP-SI": properties['nucleus_eccentricity_AP-SI'],
                        "nucleus_eccentricity_RL-SI": properties['nucleus_eccentricity_RL-SI'],
                        "nucleus_solidity": properties['nucleus_solidity'],
                        "nucleus_volume": properties['nucleus_volume'],
                        "nucleus_median_thickness": properties['nucleus_median_thickness'],
                        "intensity_variation": properties['intensity_variation'],
                        "median_thickness": properties['median_thickness'],
                        "center": properties['center'],
                        "volume": properties['volume']
                    }
                    rows.append(row)
    metrics['discs'] = rows

    return metrics, imgs

def measure_disc(img_data, seg_disc_data, centerline, csf_signal, pr):
    '''
    Calculate metrics from binary disc segmentation
    '''
    # Fetch coords from image
    coords = np.argwhere(seg_disc_data > 0)
    
    # Exclude discs touching image boundary
    if coords[:,2].max() == seg_disc_data.shape[2]-1 or coords[:,2].min() == 0:
        return None, None, False

    # Find closest point and derivative onto the centerline
    z_mean = np.mean(coords, axis=0)[-1]
    closest_centerline_idx = np.argmin(abs(centerline['position'][2]-z_mean))
    centerline_pos, centerline_deriv = centerline['position'][:,closest_centerline_idx], centerline['derivative'][:,closest_centerline_idx]

    # Use centerline deriv to compute discs metrics
    ellipsoid = fit_ellipsoid(coords, centerline_deriv)

    # Extract SI thickness
    bin_size = max(2//pr, 1) # Put 1 bin per 2 mm
    median_thickness = compute_thickness_profile(coords, ellipsoid['rotation_matrix'], bin_size=bin_size)

    if np.isnan(median_thickness):
        return None, None, False
    
    # Extract disc intensity in middle of the disc
    middle_RLslice = int(ellipsoid['center'][0])
    values_2d = np.array([img_data[c[0], c[1], c[2]] for c in coords if middle_RLslice -1 <= c[0] < middle_RLslice + 1])
    values_3d = np.array([img_data[c[0], c[1], c[2]] for c in coords])
    # Normalize disc intensity using CSF signal
    values_3d = values_3d / (csf_signal - min(values_2d))
    values_2d = values_2d / (csf_signal - min(values_2d))
    # peaks = find_intensity_peaks(values_smooth)
    min_peak = np.percentile(values_2d, 10) # peaks[0]
    max_peak = np.percentile(values_2d, 90) # peaks[-1]

    # Fetch shape of nucleus (max intensity region)
    nucleus_coords = np.array([c for i, c in enumerate(coords) if values_3d[i] >= max_peak])
    ellipsoid_nucl = fit_ellipsoid(np.array(nucleus_coords), centerline_deriv, min_size=3)
    nucleus_thickness = compute_thickness_profile(nucleus_coords, ellipsoid_nucl['rotation_matrix'], bin_size=bin_size)

    # Extract disc volume
    voxel_volume = pr**3
    volume = ellipsoid['volume']*voxel_volume # mm3
    nucleus_volume = ellipsoid_nucl['volume']*voxel_volume # mm3

    properties = {
        'center': np.round(ellipsoid['center']),
        'median_thickness': median_thickness*pr,
        'intensity_variation': (max_peak - min_peak),
        'volume': volume,
        'eccentricity_AP-RL': ellipsoid['eccentricity_AP-RL'],
        'eccentricity_AP-SI': ellipsoid['eccentricity_AP-SI'],
        'eccentricity_RL-SI': ellipsoid['eccentricity_RL-SI'],
        'solidity': ellipsoid['solidity'],
        'nucleus_eccentricity_AP-RL': ellipsoid_nucl['eccentricity_AP-RL'],
        'nucleus_eccentricity_AP-SI': ellipsoid_nucl['eccentricity_AP-SI'],
        'nucleus_eccentricity_RL-SI': ellipsoid_nucl['eccentricity_RL-SI'],
        'nucleus_solidity': ellipsoid_nucl['solidity'],
        'nucleus_volume': nucleus_volume,
        'nucleus_median_thickness': nucleus_thickness*pr,
    }

    # Center volume for visualization
    _, (xmin, xmax, ymin, ymax, zmin, zmax) = crop_around_binary(seg_disc_data)

    # Normalize image intensity
    p10 = np.percentile(img_data, 5)
    p90 = np.percentile(img_data, 95)
    img_disc = (img_data - p10) / (p90 - p10 + 1e-8)

    # Extract 2D cut of disc image
    padding = 8
    ymax, zmax = [v + padding if v + padding < img_disc.shape[1+i] else img_disc.shape[1+i]-1 for i, v in enumerate((ymax, zmax))]
    ymin, zmin = [v - padding if v - padding >= 0 else 0 for v in (ymin, zmin)]
    disc_img = img_disc[xmin:xmax, ymin:ymax, zmin:zmax]
    disc_img = disc_img[int((xmax-xmin)//2)]
    disc_img_bgr = np.stack([disc_img]*3, axis=-1)

    # Overlay disc segmentation on image
    disc_seg = seg_disc_data[xmin:xmax, ymin:ymax, zmin:zmax]
    disc_seg = disc_seg[int((xmax-xmin)//2)]
    disc_seg_bgr = disc_img_bgr.copy()
    disc_seg_bgr[disc_seg > 0] = [0, 0, 1] # Red overlay

    # # Overlay centerline_deriv vector in yellow onto the disc_img_bgr
    # center_y, center_z = disc_img_bgr.shape[0] // 2, disc_img_bgr.shape[1] // 2
    # # Project the centerline_deriv vector onto the YZ plane (since we are slicing along X)
    # deriv_yz = centerline_deriv[1:3]
    # deriv_yz = deriv_yz / (np.linalg.norm(deriv_yz) + 1e-8)
    # length = min(disc_img_bgr.shape[0], disc_img_bgr.shape[1]) // 3
    # end_y = int(center_y + deriv_yz[0] * length)
    # end_z = int(center_z + deriv_yz[1] * length)
    # cv2.arrowedLine(
    #     disc_img_bgr,
    #     (center_z, center_y),
    #     (end_z, end_y),
    #     color=(0, 1, 1),  # Yellow in BGR
    #     thickness=2,
    #     tipLength=0.2
    # )

    img_dict = {'seg':disc_seg_bgr, 'img':disc_img_bgr}
    return properties, img_dict, True

def measure_csf(img_data, seg_csf_data):
    '''
    Extract signal from cerebro spinal fluid (CSF)
    '''
    # Extract min and max index in Z direction
    X, Y, Z = seg_csf_data.nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    coords = np.argwhere(seg_csf_data > 0)
    values = np.array([img_data[c[0], c[1], c[2]] for c in coords])

    # Loop across z axis
    properties = {
        'slice_signal':{},
        'csf_signal': np.percentile(values, 90)
    }
    for iz in range(min_z_index, max_z_index + 1):
        # Extract csf coordinates in the slice
        slice_csf = seg_csf_data[:, :, iz].astype(bool)

        # Extract images values using segmentation
        slice_values = img_data[:, :, iz][slice_csf]

        # Extract most represented value
        if slice_values.size == 0:
            signal = 0
        else:
            signal = np.percentile(slice_values, 90)

        # Save values
        properties['slice_signal'][iz] = signal/properties['csf_signal']
    return properties

def measure_canal(seg_canal, centerline, spine_centerline):
    '''
    Based on https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/spinalcordtoolbox/process_seg.py

    Expected orientation is RPI

    Extract canal metrics using:
    - canal segmentation
    - spine segmentation
    '''
    # List properties
    property_list = [
        'area_canal',
        'area_spinalcord',
        'diameter_AP_canal',
        'diameter_AP_spinalcord',
        'diameter_RL_canal',
        'diameter_RL_spinalcord',
        'canal_centroid',
        'eccentricity_canal',
        'eccentricity_spinalcord',
        'solidity_canal',
        'solidity_spinalcord',
        'asymmetry_canal_R_L',
        'asymmetry_spinalcord_R_L'
    ]
    # Fetch dimensions from image.
    nx, ny, nz, nt, px, py, pz, pt = seg_canal.dim

    # Straighten canal and spinalcord to extract metrics
    radius = 50 # mm, radius of the straightened patch to extract around the centerline
    canal_seg = (seg_canal.data > 0).astype(int)
    sc_seg = (seg_canal.data == 1).astype(int)
    straightened_coordinates, first_z_index = straighten_coordinates(centerline, spine_centerline, radius=radius)
    straightened_canal = ndimage.map_coordinates(canal_seg, straightened_coordinates, order=1, mode='grid-constant')
    straightened_spinalcord = ndimage.map_coordinates(sc_seg, straightened_coordinates, order=1, mode='grid-constant')    

    # Loop across the S-I slices
    min_index = np.argwhere(straightened_canal>0)[:,2].min()
    max_index = np.argwhere(straightened_canal>0)[:,2].max()
    shape_properties = {key: {} for key in property_list}
    for iz in range(min_index, max_index + 1):
        # Calculate shape metrics
        shape_property = _properties2d(straightened_canal[:, :, iz], straightened_spinalcord[:, :, iz], [px, py])

        if shape_property is not None:
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][first_z_index+iz] = shape_property[property_name]
        else:
            raise ValueError(f'Warning: error with slice {first_z_index+iz}.')

    return shape_properties

def measure_vertebra(img_data, seg_vert_data, seg_canal_data, canal_centerline, pr):
    '''
    Returns:
        properties: python dict
        image: 3D numpy array representing the structure of interest
    '''
    # Extract vertebra coords
    coords = np.argwhere(seg_vert_data > 0)

    # Extract z position (SI) of the vertebra
    vert_pos = np.mean(coords,axis=0)

    # Exclude vertebrae touching image boundary
    if coords[:,2].max() == seg_vert_data.shape[2]-1 or coords[:,2].min() == 0:
        return None, None, None, False

    # # Show canal
    # for i in range(3):
    #     canal_slice = np.argmax(seg_canal_data, axis=i)
    #     canal_slice_rgb = cv2.cvtColor(
    #         canal_slice.astype(np.uint8),
    #         cv2.COLOR_GRAY2RGB
    #     )
    #     idx = [j for j in range(3) if j != i]
    #     for coords in np.round(canal_centerline['position']).astype(int).T:
    #         canal_slice_rgb[coords[idx[0]], coords[idx[1]]] = [0, 0, 1] # Red overlay
    #     os.makedirs(f'test/canal/', exist_ok=True)
    #     cv2.imwrite(f'test/canal/projection_{i}.png', canal_slice_rgb*255)

    # Find closest point and derivative onto the canal centerline
    canal_dist = np.linalg.norm(canal_centerline['position'].T - vert_pos, axis=1)
    closest_canal_idx = np.argmin(canal_dist)
    canal_pos = canal_centerline['position'][:,closest_canal_idx]
    canal_deriv = canal_centerline['derivative'][:,closest_canal_idx]

    # Create two perpendicular vectors u1 and u2
    v = canal_deriv/(np.linalg.norm(canal_deriv) + 1e-8)
    tmp = np.array([1, 0, 0]) # Init temporary non colinear vector
    u1 = np.cross(v, tmp)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2)

    # Project coords in u1u2 plane
    x_coords = np.dot(coords, u1)
    y_coords = np.dot(coords, u2)
    x_canal = np.dot(canal_pos, u1)
    y_canal = np.dot(canal_pos, u2)
    min_x, min_y = np.min(x_coords), np.min(y_coords)
    # Center the image onto the segmentation
    x_coords = x_coords - min_x
    y_coords = y_coords - min_y
    x_canal = x_canal - min_x
    y_canal = y_canal - min_y
    
    # Round coordinates
    x_coords = np.round(x_coords).astype(int)
    y_coords = np.round(y_coords).astype(int)
    x_canal = np.round(x_canal).astype(int)
    y_canal = np.round(y_canal).astype(int)

    # Create image (avoid off-by-one / negative indexing wrap)
    max_x = int(np.max(x_coords))
    max_y = int(np.max(y_coords))
    seg = np.zeros((max_x + 1, max_y + 1), dtype=np.float32)
    x_coords = np.clip(x_coords, 0, max_x)
    y_coords = np.clip(y_coords, 0, max_y)
    seg[x_coords, y_coords] = 1.0

    # Pad seg by 5 pixels
    seg = np.pad(seg, pad_width=5, mode='constant', constant_values=0)
    x_canal += 5
    y_canal += 5

    #seg[x_canal, y_canal] = 2.0 # Ensure canal point is included in the segmentation

    # Symmetry vector in projected (u1, u2) plane constrained through canal point
    _, sym_vec_2d, _ = find_symmetry_vector_binary(seg.copy(), center=(x_canal, y_canal), angle_step_deg=0.2)
    w = sym_vec_2d[0] * u1 + sym_vec_2d[1] * u2
    w /= (np.linalg.norm(w) + 1e-8)

    # # Visualize symmetry axis on projected segmentation
    # seg_u8 = (seg > 0).astype(np.uint8) * 255
    # hog_overlay = cv2.cvtColor(seg_u8, cv2.COLOR_GRAY2BGR)
    # center_row = x_canal
    # center_col = y_canal
    # line_len = 10
    # drow = np.round(sym_vec_2d[0]).astype(int) * line_len
    # dcol = np.round(sym_vec_2d[1]).astype(int) * line_len
    # #cv2.line(hog_overlay, p1, p2, color=(0, 255, 0), thickness=1)
    # cv2.circle(hog_overlay, (center_col, center_row), radius=2, color=(0, 0, 255), thickness=-1)
    # cv2.circle(hog_overlay, (center_col+dcol, center_row+drow), radius=2, color=(0, 0, 255), thickness=-1)
    # hog_overlay = hog_overlay.astype(np.float32)

    # Find function zero between 0 and pi
    if (vert_pos[1]-canal_pos[1])*w[1] < 0:
        # Orient vector from canal to body
        w = -w
    u = np.cross(v, w) # create last vector

    # Find canal distance to vertebral body
    z_mean = canal_pos[2]
    canal_slice_coords = np.argwhere(seg_canal_data[:,:,int(np.round(z_mean))]>0)
    projections = np.dot(canal_slice_coords-canal_pos[:2], w[:2])

    if len(projections) == 0:
        return None, None, None, False

    anterior_radius = projections.max()
    posterior_radius = projections.min()

    # Isolate vertebral body
    anterior_pos = np.array([canal_pos[0], canal_pos[1]+anterior_radius, canal_pos[2]])
    ant_projections = np.dot(coords-anterior_pos, w)
    body_coords = coords[ant_projections>0]
    body_pos = np.mean(body_coords,axis=0)

    if not body_coords.any():
        return None, None, None, False

    # Recreate body volume without rotation
    body_array = np.zeros_like(seg_vert_data)
    for coord in body_coords:
        body_array[coord[0], coord[1], coord[2]]=1

    # Isolate processes
    posterior_pos = np.array([canal_pos[0], canal_pos[1]+posterior_radius, canal_pos[2]])
    projections = np.dot(coords-posterior_pos, w)
    process_coords = coords[projections<0]
    process_pos = np.mean(process_coords,axis=0)

    # Fetch AP thickness
    AP_thickness = np.max(ant_projections)

    # Compute thickness profile vertebral body
    coordinate_system = np.stack((u, w, v), axis=0)
    bin_size = max(2//pr, 1) # Put 1 bin per 2 mm
    
    median_thickness = compute_thickness_profile(body_coords, coordinate_system, bin_size=bin_size)
    
    if np.isnan(median_thickness):
        return None, None, None, False
    
    # Extract vertebral body volume
    voxel_volume = pr**3
    volume = body_coords.shape[0]*voxel_volume # mm3

    # Analyze vertebrae intensity
    body_values = np.array([img_data[c[0], c[1], c[2]] for c in body_coords])
    process_values = np.array([img_data[c[0], c[1], c[2]] for c in process_coords])

    if len(body_values) == 0 or len(process_values) == 0:
        if len(body_values) == 0:
            median_signal = np.median(process_values)
        else:
            median_signal = np.median(body_values)
        ap_attenuation = 1
    else:
        median_signal = (abs(posterior_radius)*np.median(body_values) + abs(anterior_radius)*np.median(process_values)) / (abs(posterior_radius)+abs(anterior_radius)+1e-8)
        ap_attenuation = np.median(body_values)/median_signal

    properties = {
        'center': np.round(body_pos),
        'median_thickness': median_thickness*pr,
        'AP_thickness': AP_thickness*pr,
        'volume': volume,
        'median_signal': median_signal,
        'ap_attenuation': ap_attenuation
    }

    # Recreate volume for visualization
    _, (xmin, xmax, ymin, ymax, zmin, zmax) = crop_around_binary(body_array)
    
    # Normalize image intensity
    p10 = np.percentile(img_data, 5)
    p90 = np.percentile(img_data, 95)
    img_vert = (img_data - p10) / (p90 - p10 + 1e-8)

    # Extract 2D cut of disc image
    padding = 8
    ymax, zmax = [v + padding if v + padding < img_vert.shape[1+i] else img_vert.shape[1+i]-1 for i, v in enumerate((ymax, zmax))]
    ymin, zmin = [v - padding if v - padding >= 0 else 0 for v in (ymin, zmin)]
    vert_img = img_vert[xmin:xmax, ymin:ymax, zmin:zmax]
    vert_img = vert_img[int((xmax-xmin)//2)]
    vert_img_bgr = np.stack([vert_img]*3, axis=-1)

    # Overlay vertebrae segmentation on image
    vert_seg = seg_vert_data[xmin:xmax, ymin:ymax, zmin:zmax]
    vert_seg = vert_seg[int((xmax-xmin)//2)]
    vert_seg_bgr = vert_img_bgr.copy()
    vert_seg_bgr[vert_seg > 0] = [0, 0, 1] # Red overlay

    img_dict = {'seg':vert_seg_bgr, 'img':vert_img_bgr}

    return properties, img_dict, body_array, True

def measure_foramens(foramens_name, img_data, seg_foramen_data, seg_canal_data, canal_centerline, spine_centerline, pr):
    '''
    This function measures the surface of the left and right neural foramen formed by 2 vertebrae and a disc

    Parameters:
        img_data: 3D numpy array containing the image data
        seg_foramen_data: 3D numpy array containing
            - a segmentation of the top and bottom vertebrae equal to 1
            - a segmentation of the intervertebral discs in between equal to 2
        canal_centerline: python dict
        spine_centerline: python dict

    Returns:
        foramens_areas: python dict
            left and right surface of the foramina (neuroligical convention, so patient's left is radiological right)
        foramens_imgs:
            left and right image of the foramina
    '''
    # Extract vertebrae and disc coords
    foramens_coords = np.argwhere(seg_foramen_data > 0)
    canal_coords = np.argwhere(seg_canal_data > 0)

    # Extract z position (SI) of the disc center of mass
    if 2 in seg_foramen_data:
        disc_coords = np.argwhere(seg_foramen_data == 2)
        disc_pos = np.mean(disc_coords,axis=0)
    else:
        disc_pos = np.mean(foramens_coords,axis=0)

    # Find closest point using distance to the centerline
    canal_dist = np.linalg.norm(canal_centerline['position'].T - disc_pos, axis=1)
    spine_dist = np.linalg.norm(spine_centerline['position'].T - disc_pos, axis=1)
    canal_pos, canal_deriv = canal_centerline['position'][:,np.argmin(canal_dist)], canal_centerline['derivative'][:,np.argmin(canal_dist)]
    spine_pos, spine_deriv = spine_centerline['position'][:,np.argmin(spine_dist)], spine_centerline['derivative'][:,np.argmin(spine_dist)]

    # Create vector w with canal centerline and spine centerline
    v = canal_deriv/np.linalg.norm(canal_deriv)
    w = spine_pos - canal_pos
    w = w/np.linalg.norm(w)
    w[2] = (-v[0]*w[0] - v[1]*w[1]) / (v[2]+1e-8) # Make w orthogonal to v
    w = w/np.linalg.norm(w)
    
    n = np.cross(v, w) # normal vector of the plane
    n /= np.linalg.norm(n)
    
    dot_product = np.dot(foramens_coords-canal_pos, n)

    # Extract foramen reference position
    if foramens_name != "foramens_C1-C2":
        dot_product_ref = np.dot(disc_coords-canal_pos, w)
        ref_pos = disc_coords[np.argmin(dot_product_ref)]
    else:
        ref_pos = canal_pos + 2*w # Arbitrary position 2mm away from canal in the direction of w for C1-C2 foramens since there is no disc

    # Generate mask to extract image intensity
    cropped_canal_mask = seg_canal_data.astype(bool).astype(int).copy()
    foramen_cube_mask = np.zeros_like(seg_foramen_data)
    foramen_cube_mask[np.min(foramens_coords[:,0]):np.max(foramens_coords[:,0]), np.min(foramens_coords[:,1]):np.max(foramens_coords[:,1]), np.min(foramens_coords[:,2]):np.max(foramens_coords[:,2])] = 1
    cropped_canal_mask[~foramen_cube_mask.astype(bool)] = 0 # Keep only canal at foramen level
    dilated_canal_mask = fastest_dilation_edt(cropped_canal_mask, radius=10)
    canal_dilated_coords = np.argwhere((dilated_canal_mask-cropped_canal_mask) > 0)
    canal_dilated_values = img_data[canal_dilated_coords[:, 0], canal_dilated_coords[:, 1], canal_dilated_coords[:, 2]]
    canal_dilated_proj_v = np.round(np.dot(canal_dilated_coords-canal_pos, v)).astype(int) # SI*
    canal_dilated_proj_w = np.round(np.dot(canal_dilated_coords-canal_pos, w)).astype(int) # AP*
    canal_dilated_proj_n = np.round(np.dot(canal_dilated_coords-canal_pos, n)).astype(int) # RL*

    # Distinguish left-from-right
    pos_coords = dot_product>0
    neg_coords = ~pos_coords

    # Neurological orientation (patient's left is radiological right)
    halfs = {"left": foramens_coords[pos_coords], "right":foramens_coords[neg_coords]}

    # Project foramens
    foramens_areas = {}
    foramens_seg = {}
    foramens_img = {}
    for side, coords in halfs.items():
        foramen_mask = None
        # Project coords in vw plane
        x_coords = np.dot(coords-canal_pos, v)
        y_coords = np.dot(coords-canal_pos, w)
        canal_coords_x = np.dot(canal_coords-canal_pos, v)
        canal_coords_y = np.dot(canal_coords-canal_pos, w)
        ref_coord_x = np.dot(ref_pos-canal_pos, v)
        ref_coord_y = np.dot(ref_pos-canal_pos, w)

        # Center the image onto the segmentation
        min_x = np.min(x_coords)
        min_y = np.min(y_coords)
        x_coords = x_coords - min_x
        y_coords = y_coords - min_y
        canal_coords_x = canal_coords_x - min_x
        canal_coords_y = canal_coords_y - min_y
        ref_coord_x = ref_coord_x - min_x
        ref_coord_y = ref_coord_y - min_y

        # Round coordinates
        x_coords = np.round(x_coords).astype(int)
        y_coords = np.round(y_coords).astype(int)
        canal_coords_x = np.round(canal_coords_x).astype(int)
        canal_coords_y = np.round(canal_coords_y).astype(int)
        ref_coord_x = np.round(ref_coord_x).astype(int)
        ref_coord_y = np.round(ref_coord_y).astype(int)

        # Create image
        seg = np.zeros((np.max(x_coords)+1, np.max(y_coords)+1))
        for x, y in zip(canal_coords_x, canal_coords_y):
            if x > 0 and y > 0 and x < seg.shape[0] and y < seg.shape[1]:
                seg[x, y]=2
        
        for x, y in zip(x_coords, y_coords):
            seg[x, y]=1
        
        # Inverse image
        foramen_bg = morphology.remove_small_objects(~(seg==1).astype(bool), min_size=10)
        canal_bg = morphology.remove_small_objects((seg==2).astype(bool), min_size=10)

        # Padd image to connect exterior components
        padding = 70
        ref_coord_x = ref_coord_x + padding
        ref_coord_y = ref_coord_y + padding
        foramen_bin = np.pad(foramen_bg, pad_width=(padding, padding), mode='constant', constant_values=1)
        canal_bin = np.pad(canal_bg, pad_width=(padding, padding), mode='constant', constant_values=0)

        if foramens_name == "foramens_C1-C2":
            # Use spinal canal for cervical foramens (seems more robust)
            # Label all component and extract regions
            labeled_foramen, _ = ndi.label(foramen_bin)
            foramen_regions = measure.regionprops(labeled_foramen)

            labeled_canal, _ = ndi.label(canal_bin)
            canal_regions = measure.regionprops(labeled_canal)

            # Match components between canal and foramen
            match_dict = {}
            for val in np.unique(labeled_canal):
                if val != 0:
                    mask = labeled_canal == val
                    vals = np.unique(labeled_foramen[mask])
                    if len(vals) > 1:
                        raise ValueError('Error in matching canal and foramen components, multiple matches found.')
                    else:
                        match_dict[int(val)] = int(vals[0])
            
            # Save foramens
            if len(canal_regions) > 1:
                # Find region closer to spine coordinate
                if len(foramen_regions) > 1:
                    foramen_areas = [region.area for region in foramen_regions]
                    foramen_regions_nomax = [region for region in foramen_regions if region.area != np.max(foramen_areas)]
                    ref_foramen_dists = [np.sqrt((region.centroid[0]-ref_coord_x)**2 + (region.centroid[1]-ref_coord_y)**2) for region in foramen_regions_nomax]
                    sorted_foramen_regions = [region for _, region in sorted(zip(ref_foramen_dists, foramen_regions_nomax), key=lambda x: x[0])]
                    closest_canal_region = None
                    for foramen_region in sorted_foramen_regions:
                        mathing_list = [match_dict[region.label] == foramen_region.label for region in canal_regions]
                        if any(mathing_list):
                            closest_canal_region = canal_regions[mathing_list.index(True)]
                            closest_foramen_region = foramen_region
                            break
                    
                    if closest_canal_region is None:
                        # If no match is found, find closest canal region to spine coordinate
                        canal_dists = [np.sqrt((region.centroid[0]-ref_coord_x)**2 + (region.centroid[1]-ref_coord_y)**2) for region in canal_regions]
                        closest_canal_region = canal_regions[np.argmin(canal_dists)]
                        foramen_mask = labeled_canal == closest_canal_region.label
                    else:
                        if closest_foramen_region.area <= 2*closest_canal_region.area:
                            # Check if there is a match in region size
                            foramen_mask = labeled_foramen == closest_foramen_region.label
                        else:
                            foramen_mask = labeled_canal == closest_canal_region.label
                else:
                    canal_dists = [np.sqrt((region.centroid[0]-ref_coord_x)**2 + (region.centroid[1]-ref_coord_y)**2) for region in canal_regions]
                    closest_canal_region = canal_regions[np.argmin(canal_dists)]
                    foramen_mask = labeled_canal == closest_canal_region.label              

                # Calculate foramen area
                pixel_surface = pr**2
                foramen_area = np.argwhere(foramen_mask > 0).shape[0]*pixel_surface #mm2
                foramens_areas[side] = foramen_area
                
                # Flip the foraminal image upside-down for better visual
                img = np.flipud(foramen_bin) + np.flipud(foramen_mask.astype(int))

                # Remove padding
                img_padding = padding - 5
                foramens_seg[side] = img[img_padding:-img_padding, img_padding:-img_padding]
            else:
                foramens_areas[side] = -1
                img = np.flipud(foramen_bin)
                # Remove padding
                img_padding = padding - 5
                foramens_seg[side] = img[img_padding:-img_padding, img_padding:-img_padding]
        else:
            foramen_masks_list = []
            for i in range(8):
                # Dilate foramen mask and find largest connected component to extract foramen area
                foramen_bin_dilate = morphology.binary_dilation(~foramen_bin, morphology.disk(i))
                foramen_bin_dilate_no_small = ~morphology.remove_small_objects(~foramen_bin_dilate, min_size=2)
                labeled_foramen, _ = ndi.label(~foramen_bin_dilate_no_small)
                foramen_regions = measure.regionprops(labeled_foramen)
                if len(foramen_regions) > 1:
                    foramen_areas = [region.area for region in foramen_regions]
                    foramen_regions_nomax = [region for region in foramen_regions if region.area != np.max(foramen_areas)]
                    ref_foramen_dists = [np.sqrt((region.centroid[0]-ref_coord_x)**2 + (region.centroid[1]-ref_coord_y)**2) for region in foramen_regions_nomax]
                    sorted_foramen_regions = [region for _, region in sorted(zip(ref_foramen_dists, foramen_regions_nomax), key=lambda x: x[0])]
                    sorted_eroded_masks = [labeled_foramen == region.label for region in sorted_foramen_regions]
                    sorted_dilated_masks = [~morphology.binary_erosion(~eroded_mask, morphology.disk(i)) for eroded_mask in sorted_eroded_masks] # Dilate to the original shape
                    if len(sorted_dilated_masks) > 1:
                        foramen_masks_list.append(sorted_dilated_masks[:2]) # Keep only the 2 closest regions to the ref coordinate
                    else:
                        foramen_masks_list.append(sorted_dilated_masks)
            if len(foramen_masks_list) != 0:
                concatenated_foramen_masks_list = [mask for sublist in foramen_masks_list for mask in sublist]
                list_confidence_vector = [foramen_confidence_score(mask, concatenated_foramen_masks_list) for mask in concatenated_foramen_masks_list]
                closest_foramen_short_list = [mask for mask in concatenated_foramen_masks_list if foramen_confidence_score(mask, concatenated_foramen_masks_list) == np.max(list_confidence_vector)]
                if np.max(list_confidence_vector) == 1:
                    area = [np.sum(mask) for mask in closest_foramen_short_list]
                    foramen_mask = closest_foramen_short_list[np.argmax(area)]
                else:
                    foramen_mask = closest_foramen_short_list[0] # Choose the first mask
                # Calculate foramen area
                pixel_surface = pr**2
                foramen_area = np.argwhere(foramen_mask > 0).shape[0]*pixel_surface #mm2
                foramens_areas[side] = foramen_area
                
                # Flip the foraminal image upside-down for better visual
                img = np.flipud(foramen_bin) + np.flipud(foramen_mask.astype(int))

                # Remove padding
                img_padding = padding - 5
                foramens_seg[side] = img[img_padding:-img_padding, img_padding:-img_padding]
            else:
                foramens_areas[side] = -1
                img = np.flipud(foramen_bin)
                # Remove padding
                img_padding = padding - 5
                foramens_seg[side] = img[img_padding:-img_padding, img_padding:-img_padding]
                foramen_mask = None
        
        foramens_img[side] = np.zeros(foramens_seg[side].shape, dtype=np.float32)
        ## Extract foramen intensity from extruded 2D foramen mask intersected with canal shell
        if foramen_mask is None or np.sum(foramen_mask) == 0 or canal_dilated_coords.shape[0] == 0:
            continue

        # Remove max padding
        foramen_mask_no_pad = foramen_mask[padding:-padding, padding:-padding]
        foramen_mask_3d = np.array([foramen_mask_no_pad]*img_data.shape[0])

        # Extract projected mask coords
        foramen_coords_3d = np.argwhere(foramen_mask_3d > 0)

        if side == "left":
            foramen_coords_3d[:, 0] = np.round(foramen_coords_3d[:, 0]).astype(int) # n
        else:
            foramen_coords_3d[:, 0] = np.round(-foramen_coords_3d[:, 0]).astype(int) # n
        foramen_coords_3d[:, 1] = np.round(foramen_coords_3d[:, 1] + min_x).astype(int) # v
        foramen_coords_3d[:, 2] = np.round(foramen_coords_3d[:, 2] + min_y).astype(int) # w

        # Center the image onto the segmentation
        min_v = np.min(np.concatenate([foramen_coords_3d[:, 1], canal_dilated_proj_v], axis=0))
        min_w = np.min(np.concatenate([foramen_coords_3d[:, 2], canal_dilated_proj_w], axis=0))
        min_n = np.min(np.concatenate([foramen_coords_3d[:, 0], canal_dilated_proj_n], axis=0))
        foramen_coords_3d[:, 1] = foramen_coords_3d[:, 1] - min_v
        foramen_coords_3d[:, 2] = foramen_coords_3d[:, 2] - min_w
        foramen_coords_3d[:, 0] = foramen_coords_3d[:, 0] - min_n
        canal_dilated_proj_v_side = canal_dilated_proj_v - min_v
        canal_dilated_proj_w_side = canal_dilated_proj_w - min_w
        canal_dilated_proj_n_side = canal_dilated_proj_n - min_n
        max_v = np.max(np.concatenate([foramen_coords_3d[:, 1], canal_dilated_proj_v_side], axis=0)) # SI
        max_w = np.max(np.concatenate([foramen_coords_3d[:, 2], canal_dilated_proj_w_side], axis=0)) # AP
        max_n = np.max(np.concatenate([foramen_coords_3d[:, 0], canal_dilated_proj_n_side], axis=0)) # RL

        seg_3d_foramen = np.zeros((max_n+1, max_v+1, max_w+1), dtype=np.uint8) # RL, AP, SI
        for x, y, z in zip(foramen_coords_3d[:, 0], foramen_coords_3d[:, 1], foramen_coords_3d[:, 2]):
            if x > 0 and y > 0 and x < seg_3d_foramen.shape[0] and y < seg_3d_foramen.shape[1] and z < seg_3d_foramen.shape[2]:
                seg_3d_foramen[x, y, z] = 1
        
        seg_3d_canal = np.zeros((max_n+1, max_v+1, max_w+1), dtype=np.float32) # RL, AP, SI
        img_3d_foramen = np.zeros((max_n+1, max_v+1, max_w+1), dtype=np.float32) # RL, AP, SI
        for val, x, y, z in zip(canal_dilated_values, canal_dilated_proj_n_side, canal_dilated_proj_v_side, canal_dilated_proj_w_side):
            if x > 0 and y > 0 and x < seg_3d_canal.shape[0] and y < seg_3d_canal.shape[1] and z < seg_3d_canal.shape[2]:
                seg_3d_canal[x, y, z] = 1
                img_3d_foramen[x, y, z] = val

        mask_intersection = seg_3d_foramen * seg_3d_canal
        if np.sum(mask_intersection) == 0:
            continue
        img_3d_foramen_intersection = img_3d_foramen * mask_intersection
        median_idx = int(np.median(np.argwhere(mask_intersection!=0)[:,0]))

        if side == 'left':
            img_slice = np.sum(img_3d_foramen_intersection[median_idx:], axis=0)
        else:
            img_slice = np.sum(img_3d_foramen_intersection[:median_idx], axis=0)

        # Crop around the foramen
        nonzero = np.argwhere(img_slice!=0)
        # low_signal = np.percentile(inter_vals, 20)
        # compression_fraction = np.argwhere(img_slice>=low_signal).shape[0]/nonzero.shape[0]
        # im = (im - mi) / (ma - mi + 1e-8)
        crop_size = (np.min(nonzero[:,0]), np.max(nonzero[:,0]), np.min(nonzero[:,1]), np.max(nonzero[:,1]))
        im = np.flipud(img_slice[crop_size[0]:crop_size[1], crop_size[2]:crop_size[3]])
        shape = foramens_img[side].shape
        foramens_img[side][shape[0]//2-im.shape[0]//2:shape[0]//2+im.shape[0]-im.shape[0]//2, shape[1]//2-im.shape[1]//2:shape[1]//2+im.shape[1]-im.shape[1]//2] = im
    return foramens_areas, foramens_img, foramens_seg

def compute_foramen_compression_ratio(metric_rows, imgs):
    foramens_list = [v for k,v in imgs.items() if k.startswith("foramens") and k.endswith("img")]
    foramens_name = [k for k,v in imgs.items() if k.startswith("foramens") and k.endswith("img")]
    min_signal_list = [np.percentile(foramen_img[foramen_img!=0], 30) if np.sum(foramen_img) > 0 else 0 for foramen_img in foramens_list]
    min_signal = np.median(min_signal_list)
    compression_ratio = [(foramen_img>min_signal).sum()/(foramen_img!=0).sum() if np.sum(foramen_img) > 0 else 1 for foramen_img in foramens_list]
    rows_name = [row['name'] for row in metric_rows]
    for i, name in enumerate(foramens_name):
        foramen_name = "_".join(name.split("_")[:2])
        metric_row_idx = rows_name.index(foramen_name)
        side = name.split("_")[2]
        metric_rows[metric_row_idx][f"{side}_compression_ratio"] = compression_ratio[i]
        # normalize foramen image for visualization
        if np.sum(foramens_list[i]) > 0:
            im = foramens_list[i]
            ma = np.max(im)
            mi = np.min(im)
            im = (im - mi) / (ma - mi + 1e-8)
            imgs[name] = im
    return metric_rows, imgs

def foramen_confidence_score(mask, mask_list):
    return np.sum([2*np.sum((mask * mask_i))/(np.sum(mask_i)+np.sum(mask)) > 0.60 for mask_i in mask_list])

def find_intensity_peaks(values):
    '''
    Find intensity peaks in the histogram of values using a gaussian mixture model

    Parameters:
        values: (N,) array of intensity values corresponding to the coords

    Returns:
        peaks: list of floats representing the intensity peaks
    '''

    # Fit Gaussian Mixture Model with 2 components
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(values.reshape(-1, 1))

    # Extract means and weights of the components
    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()

    # Sort components by their means
    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    weights = weights[sorted_indices]

    # Filter out components with very low weight
    weight_threshold = 0.05
    significant_means = means[weights > weight_threshold]

    return significant_means.tolist()

def fit_ellipsoid(coords, centerline_deriv, min_size=32):
    # Compute the center of mass of the disc
    center = coords.mean(axis=0)

    # Center the coordinates
    coords_centered = coords - center
    volume = coords.shape[0]

    # Create two perpendicular vectors u1 and u2
    v = centerline_deriv / np.linalg.norm(centerline_deriv)  # Normalize the vector
    u1 = np.array([0, 1, -(v[1]/v[2])]) # AP non colinear vector
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(u1, v)
    u2 /= np.linalg.norm(u2)
    rotation_matrix = np.stack((u2, u1, v), axis=0)

    # Compute solidity and eccentricity
    def _proj_props(a, b, min_size=min_size):
        # Project coords onto plane defined by vectors a and b
        p_a = np.dot(coords_centered, a)
        p_b = np.dot(coords_centered, b)
        # Center to positive coordinates and round
        p_a = p_a - p_a.min()
        p_b = p_b - p_b.min()
        ia = np.round(p_a).astype(int)
        ib = np.round(p_b).astype(int)
        # Build binary image
        H = ia.max()
        W = ib.max()
        seg2d = np.zeros((H+1, W+1), dtype=bool)
        for x, y in zip(ia, ib):
            if x > 0 and y > 0 and x-1 < H and y-1 < W:
                seg2d[x, y] = True
        
        # Pad image
        seg2d = np.pad(seg2d, pad_width=5, mode='constant', constant_values=0)
        seg2d = morphology.remove_small_objects(seg2d, min_size=min_size).astype(int)
        props = measure.regionprops(measure.label(seg2d))
        if len(props) == 0:
            return -1.0, -1.0, -1.0
        # choose largest region if multiple
        areas = [p.area for p in props]
        region = props[np.argmax(areas)]
        # handle Darwin bug as in other places
        if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
            solidity_val = -1.0
        else:
            solidity_val = float(region.solidity)
        ecc_val = float(region.eccentricity)
        return seg2d, solidity_val, ecc_val

    # Compute 2D projection properties first
    seg_u1u2, solidity_u1u2, eccentricity_u1u2 = _proj_props(u1, u2)
    seg_u1v, solidity_u1v, eccentricity_u1v = _proj_props(u1, v)
    seg_u2v, solidity_u2v, eccentricity_u2v = _proj_props(u2, v)

    # Compute 3D solidity: ratio of object volume (voxel count) to convex hull volume
    # Need at least 4 non-coplanar points to build a 3D hull
    if coords_centered.shape[0] >= 4:
        try:
            hull = ConvexHull(coords_centered)
            hull_vol = float(getattr(hull, "volume", 0.0))
            if hull_vol > 1e-8:
                solidity_3d = float(volume) / hull_vol
            else:
                solidity_3d = 1.0
        except Exception:
            solidity_3d = 1.0
    else:
        solidity_3d = 1.0

    # Results
    ellipsoid = {
        'center': center,
        'rotation_matrix': rotation_matrix,
        'eccentricity_AP-RL': eccentricity_u1u2,
        'eccentricity_AP-SI': eccentricity_u1v,
        'eccentricity_RL-SI': eccentricity_u2v,
        'solidity': solidity_3d,
        'volume': volume
    }
    return ellipsoid

def compute_thickness_profile(coords, rotation_matrix, bin_size=1.0):
    """
    Measure thickness profile of the segmentation by splitting RL-AP plane into bins.
    
    Parameters:
        coords: (N, 3) array of 3D points of the segmentations
        rotation_matrix: (3, 3) array corresponding to new coordinate system
        bin_size: RL-AP plane resolution for thickness extraction (in voxels)

    Returns:
        median thicknesses: thickness in the RL-AP plane
    """
    # Project voxel coordinates onto the axis
    center = np.mean(coords,axis=0) 
    coords_centered = coords - center

    # Rotate coords_centered
    eigvecs = rotation_matrix
    rot_coords = coords_centered @ np.linalg.inv(eigvecs)

    # Find min and max dimensions of the disc in the RL-AP plane
    min_RL, max_RL = rot_coords[:,0].min(), rot_coords[:,0].max()
    min_AP, max_AP = rot_coords[:,1].min(), rot_coords[:,1].max()

    # Pad min and max to reduce effect of discs edges
    min_RL += 1*bin_size
    max_RL -= 1*bin_size
    min_AP += 1*bin_size
    max_AP -= 1*bin_size

    # Create bin matrix along RL and AP dimension
    bins_RL = np.arange(min_RL, max_RL + bin_size, bin_size)
    bin_indices_RL = np.digitize(rot_coords[:,0], bins_RL) - 1
    bins_AP = np.arange(min_AP, max_AP + bin_size, bin_size)
    bin_indices_AP = np.digitize(rot_coords[:,1], bins_AP) - 1

    # Fetch median thickness
    thicknesses = []
    for x in range(len(bins_RL) - 1):
        slice_mask_RL = bin_indices_RL == x
        for y in range(len(bins_AP) - 1):
            slice_mask_AP = bin_indices_AP == y
            slice_mask = slice_mask_RL*slice_mask_AP
            if any(slice_mask):
                # Extract voxels in the square
                slice_coords = rot_coords[slice_mask]

                # Find max and minimum in square
                min_SI, max_SI = slice_coords[:,2].min(), slice_coords[:,2].max()

                # Extract thickness
                thicknesses.append(max_SI-min_SI)
    return np.median(np.array(thicknesses))

def crop_around_binary(volume):
    """
    Crop a 3D numpy array around the non-zero region and return the cropped size.

    Args:
        volume : np.ndarray
            3D binary numpy array (bool or 0/1 values).

    Returns:
        cropped : np.ndarray
            Cropped 3D volume.
        bbox : tuple
            Bounding box coordinates.
    """
    assert volume.ndim == 3, "Input must be a 3D array"

    # Find non-zero coordinates
    coords = np.argwhere(volume)
    if coords.size == 0:
        return (volume.copy(), None)

    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)
    cropped = volume[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

    return cropped, (xmin, xmax, ymin, ymax, zmin, zmax)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == '__main__':
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/input/sub-001_ses-A_acq-isotropic_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/step2_output/sub-001_ses-A_acq-isotropic_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/step1_levels/sub-001_ses-A_acq-isotropic_T2w.nii.gz'

    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/input/sub-039_acq-lowresSag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step2_output/sub-039_acq-lowresSag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step1_levels/sub-039_acq-lowresSag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/input/sub-251_acq-lowresSag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step2_output/sub-251_acq-lowresSag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step1_levels/sub-251_acq-lowresSag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/input/sub-237_acq-lowresSag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step2_output/sub-237_acq-lowresSag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step1_levels/sub-237_acq-lowresSag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/input/sub-088_acq-lowresSag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step2_output/sub-088_acq-lowresSag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/spider_output/step1_levels/sub-088_acq-lowresSag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/input/sub-nMRI010_ses-Pre_acq-sagittalStirirfse_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/step2_output/sub-nMRI010_ses-Pre_acq-sagittalStirirfse_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/step1_levels/sub-nMRI010_ses-Pre_acq-sagittalStirirfse_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/input/sub-nMRI010_ses-Post2_acq-sagittalStir_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/step2_output/sub-nMRI010_ses-Post2_acq-sagittalStir_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/step1_levels/sub-nMRI010_ses-Post2_acq-sagittalStir_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/input/sub-009_acq-sag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step2_output/sub-009_acq-sag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step1_levels/sub-009_acq-sag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/input/sub-145_acq-sag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step2_output/sub-145_acq-sag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step1_levels/sub-145_acq-sag_T2w.nii.gz'
    
    img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/input/sub-060_acq-sag_T2w_0000.nii.gz'
    seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step2_output/sub-060_acq-sag_T2w.nii.gz'
    label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step1_levels/sub-060_acq-sag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/input/sub-145_acq-sag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step2_output/sub-145_acq-sag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/step1_levels/sub-145_acq-sag_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/input/sub-nMRI035_ses-Pre_acq-sagStir_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/step2_output/sub-nMRI035_ses-Pre_acq-sagStir_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/lbp_sag_out/step1_levels/sub-nMRI035_ses-Pre_acq-sagStir_T2w.nii.gz'
    
    # img_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/jacob-cervical/out/input/ESF_Post_Sag_T2w_0000.nii.gz'
    # seg_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/jacob-cervical/out/step2_output/ESF_Post_Sag_T2w.nii.gz'
    # label_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/jacob-cervical/out/step1_levels/ESF_Post_Sag_T2w.nii.gz'

    ofolder_path = 'test'

    # Load totalspineseg mapping
    resources_path = importlib.resources.files(resources)
    mapping_path = os.path.join(resources_path, 'labels_maps/tss_map.json')
    with open(mapping_path, 'r') as file:
        mapping = json.load(file)
    
    # Run measure_seg
    _measure_seg(img_path, seg_path, label_path, ofolder_path, mapping)