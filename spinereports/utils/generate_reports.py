import importlib
import re
import os, argparse, textwrap
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import copy
from tqdm import tqdm
import totalspineseg.resources as tss_resources
import spinereports.resources as sr_resources
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from matplotlib.ticker import MaxNLocator


def _font_scale_for_grid(page_size, nrows: int, ncols: int, base_cell_in: float = 4.5) -> float:
    """Return a scale factor for fonts based on page size and grid density.

    The intent is to keep text visually consistent across reports even when the
    page size and the subplot grid change (e.g., many rows/cols).
    """
    page_w, page_h = page_size
    nrows = max(1, int(nrows))
    ncols = max(1, int(ncols))
    cell_min_in = min(page_w / ncols, page_h / nrows)
    scale = cell_min_in / float(base_cell_in)
    # Clamp to avoid unreadably small or comically large text.
    return float(np.clip(scale, 0.6, 2.0))


def _fs(base_points: float, scale: float, min_fs: int = 8, max_fs: int = 120) -> int:
    """Scaled fontsize in points (matplotlib fontsize units)."""
    return int(np.clip(base_points * float(scale), min_fs, max_fs))


def _apply_report_grid_layout(
    fig: plt.Figure,
    *,
    scale: float,
    rotated_xticks: bool,
):
    """Apply consistent spacing so plots use page width and ticks don't overlap."""
    scale = float(scale) if scale else 1.0
    inv = max(0.0, (1.0 / max(scale, 1e-6)) - 1.0)

    # Use nearly full page width, reserve bottom for tick labels.
    left = 0.015
    right = 0.995
    bottom = 0.07 + (0.07 if rotated_xticks else 0.03) + 0.02 * inv
    bottom = float(np.clip(bottom, 0.08, 0.22))
    top = 0.90

    # Increase spacing when cells get small.
    wspace = float(np.clip(0.28 + 0.20 * inv, 0.28, 0.70))
    hspace = float(np.clip(0.40 + 0.30 * inv + (0.18 if rotated_xticks else 0.0), 0.40, 1.10))

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)


def _apply_report_outer_margins(fig: plt.Figure, *, scale: float, rotated_xticks: bool):
    """Apply margins without overriding GridSpec spacing."""
    scale = float(scale) if scale else 1.0
    inv = max(0.0, (1.0 / max(scale, 1e-6)) - 1.0)

    left = 0.015
    right = 0.995
    bottom = 0.07 + (0.07 if rotated_xticks else 0.03) + 0.02 * inv
    bottom = float(np.clip(bottom, 0.08, 0.24))
    top = 0.90
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _axes_with_big_header_row(
    fig: plt.Figure,
    *,
    nrows: int,
    ncols: int,
    header_height: float = 1.9,
    header_wspace: float = 0.05,
    body_wspace: float = 0.28,
    body_hspace: float = 0.55,
    outer_hspace: float = 0.06,
):
    """Create axes in row-major order with a bigger first row.

    This keeps image aspect ratio intact, while making the header row axes
    larger (both height and effective width via smaller header wspace).
    Returns a 1D numpy array of axes ordered like plt.subplots(...).flatten().
    """
    nrows = int(nrows)
    ncols = int(ncols)
    if nrows < 1 or ncols < 1:
        raise ValueError('nrows/ncols must be >= 1')

    if nrows == 1:
        gs_header = fig.add_gridspec(1, ncols, wspace=header_wspace)
        return np.array([fig.add_subplot(gs_header[0, c]) for c in range(ncols)])

    outer = fig.add_gridspec(2, 1, height_ratios=[header_height, nrows - 1], hspace=outer_hspace)
    gs_header = outer[0].subgridspec(1, ncols, wspace=header_wspace)
    gs_body = outer[1].subgridspec(nrows - 1, ncols, wspace=body_wspace, hspace=body_hspace)

    axes = []
    for c in range(ncols):
        axes.append(fig.add_subplot(gs_header[0, c]))
    for r in range(nrows - 1):
        for c in range(ncols):
            axes.append(fig.add_subplot(gs_body[r, c]))
    return np.array(axes)


def _compute_report_page_size(subject_data, all_values_df_group, metrics_dict, resources_path):
    """Compute a uniform (w,h) page size for all pages in one report group."""
    # Match the sizing heuristics used in the original plotting code.
    size_rules = {
        'canal': {'col_offset': 1, 'w': 6, 'h': 4},
        'foramens': {'col_offset': 2, 'w': 6, 'h': 6},
        'discs': {'col_offset': 4, 'w': 9, 'h': 7},
        'vertebrae': {'col_offset': 3, 'w': 6, 'h': 4},
    }

    max_w = 11.69
    max_h = 8.27
    for struc, rule in size_rules.items():
        if struc not in subject_data or struc not in all_values_df_group:
            continue
        struc_names = np.array(list(subject_data[struc].keys()))
        struc_names = struc_names[np.isin(struc_names, list(all_values_df_group[struc].keys()))].tolist()
        metrics = metrics_dict[struc]
        nrows = len(struc_names) + 1
        ncols = len(metrics) + rule['col_offset']
        max_w = max(max_w, rule['w'] * ncols)
        max_h = max(max_h, rule['h'] * nrows)

    # Make figures wider and leave room for spacing/ticks.
    # Extra width helps canal vertebra-label readability.
    return (max_w * 1.18, max_h * 1.03)


def _pdf_add_cover_page(pdf: PdfPages, page_size, subject_name: str, group: str, subject_img: str):

    def _safe_imread(path: Path):
        try:
            return plt.imread(str(path))
        except Exception:
            return None

    fig = plt.figure(figsize=page_size)
    header_ax = fig.add_axes([0.04, 0.86, 0.92, 0.12])
    header_ax.axis('off')
    img_ax = fig.add_axes([0.04, 0.06, 0.92, 0.78])
    img_ax.axis('off')

    # Scale cover fonts with page size (relative to A4 landscape).
    base_w, base_h = 11.69, 8.27
    cover_scale = float(np.clip(min(page_size[0] / base_w, page_size[1] / base_h), 0.7, 2.2))
    header_ax.text(0.00, 1.00, 'SpineReports', fontsize=_fs(130, cover_scale, min_fs=16, max_fs=150), fontweight='bold', ha='left', va='top')
    header_ax.text(0.00, 0.60, f"Subject: {subject_name}", fontsize=_fs(100, cover_scale, min_fs=11, max_fs=110), ha='left', va='top')
    header_ax.text(0.00, 0.05, f"Reference group: {group}", fontsize=_fs(90, cover_scale, min_fs=10, max_fs=100), ha='left', va='bottom', color='gray')
    header_ax.text(1.00, 1.00, datetime.now().strftime('%Y-%m-%d %H:%M'), fontsize=_fs(100, cover_scale, min_fs=8, max_fs=110), ha='right', va='top', color='gray')

    subject_img_path = Path(subject_img) if subject_img else None
    if subject_img_path and subject_img_path.exists():
        img = _safe_imread(subject_img_path)
        if img is not None:
            img_ax.imshow(img)
    else:
        img_ax.text(
            0.5,
            0.5,
            'Subject sagittal overlay image not found.',
            ha='center',
            va='center',
            fontsize=_fs(14, cover_scale, min_fs=10, max_fs=40),
            color='gray',
            transform=img_ax.transAxes,
        )

    pdf.savefig(fig)
    plt.close(fig)


def _save_individual_figure(fig: plt.Figure, images_dir: Path, stem: str):
    images_dir.mkdir(parents=True, exist_ok=True)
    # Vector version (best quality for plots/text)
    fig.savefig(str(images_dir / f"{stem}.pdf"), bbox_inches='tight')

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes the outputs of totalspineseg/utils/measure_seg.py to generate a reports.
            It requires files to follow the BIDS naming conventions. 
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            totalspineseg_generate_reports -t test_group_folder -c control_group_folder -o reports
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--test-dir', '-t', type=Path, required=True,
        help='The folder where the metrics of the test group are located (required).'
    )
    parser.add_argument(
        '--control-dir', '-c', type=Path, required=True,
        help='The folder where the metrics of the control group are located (required).'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where reports will be saved (required).'
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
    test_path = args.test_dir
    control_path = args.control_dir
    ofolder = args.ofolder
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            test_path = "{test_path}"
            control_path = "{control_path}"
            ofolder = "{ofolder}"
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    generate_reports(
        test_path=test_path,
        control_path=control_path,
        ofolder_path=ofolder,
        max_workers=max_workers,
        quiet=quiet
    )

def generate_reports(
        test_path,
        control_path,
        ofolder_path,
        max_workers,
        quiet
    ):
    # Load paths
    test_path = Path(test_path)
    control_path = Path(control_path)
    ofolder_path = Path(ofolder_path)

    # Load test demographics if exists
    if os.path.exists(str(test_path / "demographics.tsv")):
        df = pd.read_csv(str(test_path / "demographics.tsv"), sep='\t')
        demographics_test = {df.participant_id.iloc[i]:df_to_dict(df[df.participant_id == df.participant_id.iloc[i]]) for i in range(len(df))}
    else:
        demographics_test = {}
    
    # Load control demographics if exists
    if os.path.exists(str(control_path / "demographics.tsv")):
        df = pd.read_csv(str(control_path / "demographics.tsv"), sep='\t')
        demographics_control = {df.participant_id.iloc[i]:df_to_dict(df[df.participant_id == df.participant_id.iloc[i]]) for i in range(len(df))}
    else:
        demographics_control = {}
    
    # Load totalspineseg resources path
    tss_resources_path = importlib.resources.files(tss_resources)
    with open(os.path.join(tss_resources_path, 'labels_maps/levels_maps.json'), 'r') as f:
        mapping = json.load(f)
        rev_mapping = {v: k for k, v in mapping.items()}
    
    # Extract metrics values of the control group
    all_values = {'all':{}}
    subjects = [s for s in os.listdir(control_path) if os.path.isdir(control_path / s)]
    if not quiet: print("\n" "Processing control subjects:")
    for subject in tqdm(subjects, disable=quiet):
        control_sub_folder = control_path / subject
        sub_name = subject.split('_')[0]

        # Fetch demographics if available
        sex_key = None
        age_key = None
        if demographics_control and sub_name in demographics_control:
            if demographics_control[sub_name]["sex"] in ['M', 'F']:
                sex_key = f'sex_{demographics_control[sub_name]["sex"]}'
            try:
                float_age = float(demographics_control[sub_name]["age"])
            except ValueError:
                float_age = None
            if float_age is not None:
                age_key = f'age_{categorize_age_groups(float_age)}'                

        if sex_key is not None and not sex_key in all_values:
            all_values[sex_key] = {}
        if age_key is not None and not age_key in all_values:
            all_values[age_key] = {}

        # Compute metrics subject
        control_data = compute_metrics_subject(control_sub_folder)

        # Gather all values for each metric and structures
        for struc in control_data.keys():
            for struc_name in control_data[struc].keys():
                for metric in control_data[struc][struc_name].keys():
                    # Add subject to all_values
                    subject_value = control_data[struc][struc_name][metric]
                    if subject_value != -1:
                        if struc not in all_values['all']:
                            all_values['all'][struc] = {}
                        if struc_name not in all_values['all'][struc]:
                            all_values['all'][struc][struc_name] = {}
                        if metric not in all_values['all'][struc][struc_name]:
                            all_values['all'][struc][struc_name][metric] = []
                        
                        if sex_key is not None:
                            if struc not in all_values[sex_key]:
                                all_values[sex_key][struc] = {}
                            if struc_name not in all_values[sex_key][struc]:
                                all_values[sex_key][struc][struc_name] = {}
                            if metric not in all_values[sex_key][struc][struc_name]:
                                all_values[sex_key][struc][struc_name][metric] = []

                        if age_key is not None:
                            if struc not in all_values[age_key]:
                                all_values[age_key][struc] = {}
                            if struc_name not in all_values[age_key][struc]:
                                all_values[age_key][struc][struc_name] = {}
                            if metric not in all_values[age_key][struc][struc_name]:
                                all_values[age_key][struc][struc_name][metric] = []

                        all_values['all'][struc][struc_name][metric].append(subject_value)
                        if sex_key is not None:
                            all_values[sex_key][struc][struc_name][metric].append(subject_value)
                        if age_key is not None:
                            all_values[age_key][struc][struc_name][metric].append(subject_value)

    # Align canal and CSF for control group
    all_values, discs_gap, last_disc = rescale_canal(all_values, rev_mapping)
    
    # Create global figures for test data subjects
    if not quiet: print("\n" "Generating test group reports:")
    create_figures_mp(test_path, ofolder_path, all_values, demographics_test, rev_mapping, discs_gap, last_disc, max_workers, quiet)

def df_to_dict(df):
    idx = df['participant_id'].keys()[0]
    d = {k:v[idx] for k,v in df.to_dict().items()}
    return d

def create_figures_mp(test_path, ofolder_path, all_values, demographics_test, rev_mapping, discs_gap, last_disc, max_workers, quiet):
    # Create a list of test subject folders
    test_sub_folders = [test_path / subject for subject in os.listdir(test_path)]
    imgs_paths = [test_sub_folder / 'imgs' for test_sub_folder in test_sub_folders]
    ofolder_subjects = [ofolder_path / subject for subject in os.listdir(test_path)]

    process_map(
        partial(
            create_figures,
            all_values=all_values,
            demographics_test=demographics_test,
            rev_mapping=rev_mapping,
            discs_gap=discs_gap,
            last_disc=last_disc
        ),
        test_sub_folders,
        imgs_paths,
        ofolder_subjects,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )
    #create_figures(test_sub_folders[0], imgs_paths[0], ofolder_subjects[0], all_values, demographics_test, rev_mapping, discs_gap, last_disc)

def create_figures(sub_folder, imgs_path, ofolder_subject, all_values, demographics_test, rev_mapping, discs_gap, last_disc):
    # Load spinereports resources path
    resources_path = importlib.resources.files(sr_resources)

    # Load subject data
    subject_data = compute_metrics_subject(sub_folder)
    sub_name = sub_folder.name.split('_')[0]

    # Filtrate all_values if demographics are available
    sex_group = None
    age_group = None
    new_all_values = {'all': all_values['all']}
    if demographics_test and sub_name in demographics_test:
        age = demographics_test[sub_name]['age']
        sex = demographics_test[sub_name]['sex']
        if sex in ['M', 'F']:
            sex_group = f'sex_{sex}'
        try:
            float_age = float(age)
        except ValueError:
            float_age = None
        if float_age is not None:
            age_group = f'age_{categorize_age_groups(float_age)}'  
    
    if sex_group is not None and sex_group in all_values.keys():
        new_all_values[sex_group] = all_values[sex_group]
    if age_group is not None and age_group in all_values.keys():
        new_all_values[age_group] = all_values[age_group]
    all_values = copy.deepcopy(new_all_values)

    # Create median and std dictionary
    median_dict = {}
    new_all_values = copy.deepcopy(all_values)
    for group in all_values.keys():
        for struc in all_values[group].keys():
            if struc in ['foramens', 'discs', 'vertebrae']:
                for struc_name in all_values[group][struc].keys():
                    for metric, values in all_values[group][struc][struc_name].items():
                        median_added = False
                        if os.path.exists(os.path.join(resources_path, 'csv', f'{struc}_median_{metric}.csv')):
                            median_csv_path = os.path.join(resources_path, 'csv', f'{struc}_median_{metric}.csv')
                            df_median = pd.read_csv(median_csv_path)
                            med_dict = {name:{'median':float(med), 'std':float(std)} for name, med, std in zip(df_median[struc], df_median[metric], df_median[f'{metric}_std'])}
                            if struc_name in med_dict:
                                median_value = med_dict[struc_name]['median']
                                std_value = med_dict[struc_name]['std']
                                median_added = True
                        
                        if not median_added:
                            # Discard values at 4 times the std from the median
                            median_value = np.median(values)
                            std_value = np.std(values)
                            new_values = [v for v in values if v >= median_value - 4*std_value and v <= median_value + 4*std_value and v != -1]
                            new_all_values[group][struc][struc_name][metric] = new_values
                            median_value = np.median(new_values)
                            std_value = np.std(new_values)
                        
                        if group not in median_dict:
                            median_dict[group] = {}
                        if struc not in median_dict[group]:
                            median_dict[group][struc] = {struc_name: {metric: {'median': median_value, 'std': std_value}}}
                        if struc_name not in median_dict[group][struc]:
                            median_dict[group][struc][struc_name] = {metric: {'median': median_value, 'std': std_value}}
                        if metric not in median_dict[group][struc][struc_name]:
                            median_dict[group][struc][struc_name][metric] = {'median': median_value, 'std': std_value}
    
    # Compute discs gradings
    if 'T2w' in str(sub_folder):
        subject_data = compute_discs_gradings(subject_data, median_dict, do_grading=True)
    else:
        subject_data = compute_discs_gradings(subject_data, median_dict, do_grading=False)
    
    # Convert all_values to dataframe
    all_values_df = convert_to_df(new_all_values)

    # Rescale canal and CSF metrics
    interp_data = copy.deepcopy(subject_data)
    for struc in ['canal', 'csf']:
        for struc_name in subject_data[struc].keys():
            for metric in subject_data[struc][struc_name].keys():
                if metric in ['slice_nb', 'disc_level']:
                    continue
                interp_values, slice_interp = rescale_with_discs(subject_data[struc][struc_name]['disc_level'], subject_data[struc][struc_name][metric], rev_mapping, discs_gap, last_disc)
                interp_data[struc][struc_name][metric] = interp_values
            interp_data[struc][struc_name]['slice_interp'] = slice_interp
            # remove slice_nb and disc_level from dict
            interp_data[struc][struc_name].pop('slice_nb', None)
            interp_data[struc][struc_name].pop('disc_level', None)

    # Create figures    
    ofolder_subject.mkdir(parents=True, exist_ok=True)
    create_global_figures(interp_data, all_values_df, discs_gap, last_disc, median_dict, imgs_path, rev_mapping, ofolder_subject)

def compute_discs_gradings(subject_data, median_dict, do_grading=True):
    for group in median_dict.keys():
        for disc in subject_data['discs'].keys():
            median_height = None
            std_height = None
            if 'grading' not in subject_data['discs'][disc]:
                subject_data['discs'][disc]['grading'] = {}
            if disc in median_dict[group]['discs']:
                if 'DHI' in median_dict[group]['discs'][disc]:
                    median_height = median_dict[group]['discs'][disc]['DHI']['median']
                    std_height = np.std(median_dict[group]['discs'][disc]['DHI']['std'])
            if do_grading and median_height is not None and std_height is not None:
                disc_dhi = subject_data['discs'][disc]['DHI']
                disc_intensity = subject_data['discs'][disc]['intensity_variation']
                disc_solidity = subject_data['discs'][disc]['solidity']
                disc_nucleus_solidity = subject_data['discs'][disc]['nucleus_solidity']

                if disc_dhi == -1 or disc_intensity == -1 or disc_solidity == -1 or disc_nucleus_solidity == -1:
                    grade = 'Error'
                else:
                    # Normalize thickness with control median
                    normalized_thickness = disc_dhi / median_height

                    # Intensity grade
                    if disc_intensity <= 0.20:
                        intensity_grade = 5
                    elif disc_intensity <= 0.30:
                        intensity_grade = 4
                    elif disc_intensity <= 0.45:
                        intensity_grade = 3
                    elif disc_nucleus_solidity < 0.60:
                        intensity_grade =  2
                    elif disc_nucleus_solidity >= 0.60:
                        intensity_grade = 1
                    else:
                        intensity_grade = 0 # error
                    
                    # Thickness grade 
                    if normalized_thickness < 0.3 or disc_solidity < 0.70:
                        thickness_grade = 8
                    elif normalized_thickness < 0.6 or disc_solidity < 0.80:
                        thickness_grade = 7
                    elif normalized_thickness < 0.9 or disc_solidity < 0.85:
                        thickness_grade = 6
                    else:
                        thickness_grade = 1
                    
                    # Grade disc
                    if thickness_grade == 1:
                        grade = intensity_grade
                    elif intensity_grade == 5:
                        grade = thickness_grade
                    else: # Mixed grade or double entry
                        grade = f"{intensity_grade}/{thickness_grade}"

                subject_data['discs'][disc]['grading'][group] = grade
            elif not do_grading:
                subject_data['discs'][disc]['grading'][group] = 'N/A'
            else:
                subject_data['discs'][disc]['grading'][group] = 'Error'
    return subject_data

def convert_to_df(all_values):
    new_values = copy.deepcopy(all_values)
    for group in new_values.keys():
        for struc in new_values[group].keys():
            for struc_name in new_values[group][struc].keys():
                # Convert dict to dataframe with keys as columns and lines as subjects
                # Prepare a dictionary where each key is a metric and each value is a list of values for all subjects
                for i, metric in enumerate(all_values[group][struc][struc_name].keys()):
                    if metric not in ['discs_gap', 'slice_interp']:
                        data = {'subjects' : [], 'values' : [], 'slice_interp' : []}
                        for j, subject_value in enumerate(all_values[group][struc][struc_name][metric]):
                            if isinstance(subject_value, list):
                                for value, slice_interp in zip(subject_value, all_values[group][struc][struc_name]['slice_interp'][j]):
                                    data['values'].append(value)
                                    data['slice_interp'].append(slice_interp)
                                    data['subjects'].append(f'subject_{j}')
                            else:
                                data['values'].append(subject_value)
                                data['subjects'].append(f'subject_{j}')
                    df = pd.DataFrame.from_dict(
                        data,
                        orient='index'
                    ).transpose()
                    new_values[group][struc][struc_name][metric] = df
    return new_values

def compute_metrics_subject(subject_folder):
    """
    Compute metrics for a single subject and return merged_data dict for global figures.

    Parameters:
        subject_folder (Path): Path to the subject's metrics folder.
        ofolder_path (Path): Path to the output folder where reports will be saved.
        quiet (bool, optional): If True, suppresses output messages. Defaults to False.

    Returns:
        dict: A dictionary containing merged metrics data for the subject.
    """
    merged_data = {}

    # List of expected CSV files
    csv_files = {
        "canal":process_canal, 
        "csf":process_csf, 
        "discs":process_discs, 
        "foramens":process_foramens, 
        "vertebrae":process_vertebrae
    }

    # Load each CSV if it exists
    for csv_file, process_func in csv_files.items():
        csv_path = subject_folder / 'csv' / f"{csv_file}.csv"
        if csv_path.exists():
            subject_data = pd.read_csv(str(csv_path))
            # Call the compute function to process the data
            merged_data[csv_file] = process_func(subject_data)
    
    # Compute discs metrics
    merged_data = compute_discs_metrics(merged_data)

    # Compute foramen metrics
    merged_data = compute_foramens_metrics(merged_data)

    # Compute vertebrae metrics
    merged_data = compute_vertebrae_metrics(merged_data)
    return merged_data

def process_canal(subject_data):
    # Convert pandas columns to lists
    canal_dict = {'canal': {}, 'spinalcord': {}, 'spinalcord/canal': {}}
    for column in subject_data.columns[2:]:
        if column not in ['canal_centroid', 'angle_AP', 'angle_RL', 'length']:
            if not 'canal' in column:
                canal_dict['spinalcord'][column.replace('_spinalcord','')] = subject_data[column].tolist()
            if not 'spinalcord' in column:
                canal_dict['canal'][column.replace('_canal','')] = subject_data[column].tolist()
    
    # Create spinalcord/canal quotient
    for key in canal_dict['spinalcord'].keys():
        if not key in ['slice_nb', 'disc_level']:
            canal_dict['spinalcord/canal'][key] = []
            for i in range(len(canal_dict['spinalcord'][key])):
                canal_value = canal_dict['canal'][key][i]
                spinalcord_value = canal_dict['spinalcord'][key][i]
                if canal_value != 0 and canal_value != -1 and spinalcord_value != -1:
                    canal_dict['spinalcord/canal'][key].append(spinalcord_value / canal_value)
                else:
                    canal_dict['spinalcord/canal'][key].append(-1)
        else:
            canal_dict['spinalcord/canal'][key] = canal_dict['spinalcord'][key]
    return canal_dict

def process_csf(subject_data):
    # Convert pandas columns to lists
    csf_dict = {'csf': {}}
    for column in subject_data.columns[2:]:
        csf_dict['csf'][column] = subject_data[column].tolist()
    return csf_dict

def process_discs(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def process_vertebrae(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def process_foramens(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_discs_metrics(data_dict):
    # Compute Disc Height Index (DHI)
    for struc_name in data_dict['discs'].keys():
        top_vertebra = struc_name.split('-')[0]
        data_dict['discs'][struc_name]['eccentricity'] = data_dict['discs'][struc_name]['eccentricity_AP-RL']
        if top_vertebra in data_dict['vertebrae']:
            # Normalize disc height with top vertebra AP_thickness
            data_dict['discs'][struc_name]['DHI'] = data_dict['discs'][struc_name]['median_thickness'] / data_dict['vertebrae'][top_vertebra]['AP_thickness']
        else:
            data_dict['discs'][struc_name]['DHI'] = -1
    return data_dict

def compute_foramens_metrics(data_dict):
    # Compute Foramen metrics
    for struc_name in data_dict['foramens'].keys():
        if data_dict['foramens'][struc_name]['right_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != 0:
            data_dict['foramens'][struc_name]['asymmetry_R-L'] = data_dict['foramens'][struc_name]['right_surface'] / data_dict['foramens'][struc_name]['left_surface']
        else:
            data_dict['foramens'][struc_name]['asymmetry_R-L'] = -1
    return data_dict

def compute_vertebrae_metrics(data_dict):
    # Compute Vertebrae metrics
    return data_dict

def rescale_canal(all_values, rev_mapping):
    '''
    Rescale subject canals and CSF based on discs z coordinates.
    '''
    new_values = copy.deepcopy(all_values)
    struc = 'canal'
    struc_name = 'canal'
    # Align all metrics for each subject using discs level as references
    disc_levels = all_values['all'][struc][struc_name]['disc_level']
    # Flatten the list of arrays and concatenate all unique values
    all_discs = np.unique(np.concatenate([np.unique(dl) for dl in disc_levels]))
    all_discs = all_discs[~np.isnan(all_discs)]

    # For each subject count slices between discs
    n_subjects = len(disc_levels)
    gap_dict = {}
    for subj_idx in range(n_subjects):
        subj_disc_level = np.array(disc_levels[subj_idx])            
        subj_valid = ~pd.isna(subj_disc_level)
        subj_disc_positions = np.where(subj_valid)[0]
        subj_disc_values = subj_disc_level[subj_valid]

        # If the number of discs doesn't match, skip this subject
        if len(subj_disc_values) < 2:
            continue
        
        # Create dict with number of slice between discs
        previous_disc = subj_disc_values[0]
        previous_pos = subj_disc_positions[0]
        for pos, disc in zip(subj_disc_positions[1:], subj_disc_values[1:]):
            if f"{previous_disc}-{disc}" not in gap_dict:
                gap_dict[f"{previous_disc}-{disc}"] = []
            gap_dict[f"{previous_disc}-{disc}"].append(pos - previous_pos)
            previous_disc = disc
            previous_pos = pos

    # Pick max for each gap between discs in gap_dict
    gap_list = []
    discs_list = []
    for k, v in gap_dict.items():
        gap_list.append(int(round(np.median(v))))
        discs_list.append(k.split('-')[0])
        discs_list.append(k.split('-')[1])
    discs_list = [int(float(v)) for v in list(np.unique(discs_list))]
    discs_gap = int(round(np.median(gap_list)))
    last_disc = rev_mapping[max(discs_list)]

    for key in all_values.keys():
        for struc in ['canal', 'csf']:
            for struc_name in all_values[key][struc].keys():
                # Rescale subjects
                add_slice_interp = True
                for metric in all_values[key][struc][struc_name].keys():
                    if metric in ['slice_nb', 'disc_level']:
                        continue
                    for subj_idx in range(len(all_values[key][struc][struc_name][metric])):
                        try:
                            interp_values, slice_interp = rescale_with_discs(all_values[key][struc][struc_name]['disc_level'][subj_idx], all_values[key][struc][struc_name][metric][subj_idx], rev_mapping, discs_gap, last_disc)
                        except TypeError as e:
                            print(f"Error rescaling subject {subj_idx} for {struc_name} metric {metric}: {e}")
                            interp_values = []
                            slice_interp = []
                        new_values[key][struc][struc_name][metric][subj_idx] = interp_values
                        if 'slice_interp' not in new_values[key][struc][struc_name]:
                            new_values[key][struc][struc_name]['slice_interp'] = []
                        if add_slice_interp:
                            new_values[key][struc][struc_name]['slice_interp'].append(slice_interp)
                    add_slice_interp = False
                # Remove slice_nb and disc_level from dict
                new_values[key][struc][struc_name].pop('slice_nb', None)
                new_values[key][struc][struc_name].pop('disc_level', None)

    return new_values, discs_gap, last_disc

def rescale_with_discs(disc_levels, metric_list, rev_mapping, gap, last_disc):
    '''
    Return rescaled metric values and corresponding slice indices using disc levels and gap information.
    '''
    # Rescale data for each metric
    subj_disc_level = np.array(disc_levels)
    subj_valid = ~pd.isna(subj_disc_level)
    subj_disc_positions = np.where(subj_valid)[0]
    subj_disc_values = subj_disc_level[subj_valid]

    # If the number of discs doesn't match, skip this subject
    if len(subj_disc_values) < 2:
        return [], []

    # Rescale each metric with linear interpolation
    values = np.array(metric_list)
    interp_values = []
    slice_interp = []
    for disc_idx, disc in enumerate(subj_disc_values):
        if disc_idx < len(subj_disc_values) - 1:
            yp = values[subj_disc_positions[disc_idx]:subj_disc_positions[disc_idx+1]]
            xp = np.linspace(0, gap-1, len(yp))
            x = np.linspace(0, gap-1, gap)
            if not -1 in yp:
                if yp.size > 0:
                    interp_func = np.interp(
                        x=x,
                        xp=xp,
                        fp=yp
                    )
                else:
                    interp_func = np.full_like(x, 0)
            else:
                interp_func = np.full_like(x, -1)
            interp_values += interp_func.tolist()

    start_disc_gap = 0
    disc = last_disc
    mapping = {v: k for k, v in rev_mapping.items()}
    if mapping[last_disc] > int(subj_disc_values[0]):
        while disc != rev_mapping[int(subj_disc_values[0])]:
            start_disc_gap += gap
            disc = previous_structure(disc)
        slice_interp += list(range(start_disc_gap, start_disc_gap + len(interp_values)))
    else: # Handle case where test subject canal longer than control
        d = rev_mapping[int(subj_disc_values[0])]
        while d != last_disc:
            start_disc_gap += gap
            d = previous_structure(d)
        interp_values = interp_values[start_disc_gap:]
        slice_interp = list(range(len(interp_values)))
    return interp_values, slice_interp

def previous_structure(structure_name):
    '''
    Return the name of the previous structure in anatomical order.
    '''
    structure_name = structure_name.strip()

    # Handle discs (L5-S1, L4-L5, ..., T9-T10)
    # and foramens (foramens_L5-S1, foramens_L4-L5, ..., foramens_T9-T10)
    foramen = False
    if '-' in structure_name:
        if structure_name.startswith('foramens_'):
            foramen = True
            structure_name = structure_name.replace('foramens_', '')
        parts = structure_name.split('-')
        if len(parts) == 2:
            # Get previous vertebra
            lower = []
            for part in parts:
                next_lower = previous_vertebra(part)
                lower.append(next_lower)
            previous_structure = "-".join(lower)
            if foramen:
                previous_structure = "foramens_" + previous_structure
            return previous_structure
        else:
            return None
        
    # Handle vertebrae (T9, T10, T11, T12, L1, L2, L3, L4, L5, S1)
    elif re.match(r'^[TLCS]\d+$', structure_name):
        return previous_vertebra(structure_name)

def previous_vertebra(vertebra):
    vertebra = vertebra.strip()
    if re.match(r'^T\d+$', vertebra):
        if int(vertebra[1:]) == 1:
            next_lower = "C7"
        else:
            next_lower = f"T{int(vertebra[1:]) - 1}"
    elif re.match(r'^L\d+$', vertebra):
        if int(vertebra[1:]) == 1:
            next_lower = "T12"
        else:
            next_lower = f"L{int(vertebra[1:]) - 1}"
    elif re.match(r'^S', vertebra):
        next_lower = "L5"
    elif re.match(r'^C\d+$', vertebra):
        if int(vertebra[1:]) == 1:
            return None
        else:
            next_lower = f"C{int(vertebra[1:]) - 1}"
    return next_lower

def create_dict_from_subject_data(subject_data):
    """
    Create a dictionary from the subject data DataFrame.

    Parameters:
        subject_data (pd.DataFrame): The subject data DataFrame.

    Returns:
        dict: A dictionary with structure names as keys and DataFrames as values.
    """
    subject_dict = {}
    for struc in subject_data.name:
        struc_dict = {}
        struc_data = subject_data[subject_data['name'] == struc]
        struc_idx = struc_data.index[0]
        for column in struc_data.columns[2:]:
            if column != 'center':
                struc_dict[column] = struc_data[column][struc_idx]
        subject_dict[struc] = struc_dict
    return subject_dict

def create_global_figures(subject_data, all_values_df, discs_gap, last_disc, median_dict, imgs_path, rev_mapping, ofolder_path):
    """
    Create global figures from the processed subjects data.

    Parameters:
        subject_data (dict): A dictionary containing the subject data.
        all_values (dict): A dictionary containing all processed metrics data for each control subject.
        imgs_path (Path): Path to the folder containing the subject images.
        ofolder_path (Path): Path to the output folder where reports will be saved.
    """
    resources_path = importlib.resources.files(sr_resources)

    metrics_dict = {
            'discs': ['median_thickness', 'DHI', 'volume', 'eccentricity', 'solidity'],
            'vertebrae': ['median_thickness', 'AP_thickness', 'volume'],
            'foramens': ['right_surface', 'left_surface', 'asymmetry_R-L'],
            'canal': ['area', 'diameter_AP', 'diameter_RL', 'eccentricity', 'solidity'],
        }

    # Prepare output subfolders
    images_dir = Path(ofolder_path) / 'images'
    files_dir = Path(ofolder_path) / 'files'
    images_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    # Save subject_data in csv file
    for struc in subject_data.keys():
        struc_rows = []
        for struc_name in subject_data[struc].keys():
            # capture slice_interp if present per structure-name
            slice_interp = subject_data[struc][struc_name].get('slice_interp', None)
            if slice_interp is not None:
                metrics = [m for m in subject_data[struc][struc_name].keys() if m != 'slice_interp']
                start_vert = last_disc.split('-')[0]
                vertebra_level_int = np.array(slice_interp)//discs_gap
                unique_levels = np.unique(vertebra_level_int)
                vertebra_levels_names = {}
                for level in unique_levels:
                    vert_name = start_vert
                    for i in range(int(level)):
                        vert_name = previous_structure(vert_name)
                    vertebra_levels_names[level] = vert_name
                vertebra_levels = [vertebra_levels_names[int(vl)] for vl in vertebra_level_int]
                for i in range(len(slice_interp)):
                    row = {
                        'structure': struc,
                        'structure_name': struc_name,
                        'slice_interp': slice_interp[i],
                        'vertebra_level': vertebra_levels[i]
                    }
                    for metric in metrics:
                        values = subject_data[struc][struc_name][metric]
                        if isinstance(values, list):
                            row[metric] = subject_data[struc][struc_name][metric][i]
                        else:
                            raise ValueError(f"Metric {metric} for structure {struc_name} is not a list as expected.")
                    struc_rows.append(row)
            else:
                row = {'structure': struc, 'structure_name': struc_name}
                metrics = [m for m in subject_data[struc][struc_name].keys()]
                for metric in metrics:
                    val = subject_data[struc][struc_name][metric]
                    # store metric values as JSON to preserve lists
                    if isinstance(val, (float, int)):
                        row[metric] = val
                struc_rows.append(row)
        csv_path = files_dir / f"{struc}_subject.csv"
        subject_df = pd.DataFrame(struc_rows)
        subject_df.to_csv(csv_path, index=False)

    structure_titles = {
        'canal': 'Spinal canal',
        'discs': 'Intervertebral discs',
        'vertebrae': 'Vertebrae',
        'foramens': 'Intervertebral foramens',
    }

    # Create a PDF report directly (avoid rasterizing figures into PNGs first)
    for group in all_values_df.keys():
        page_size = _compute_report_page_size(subject_data, all_values_df[group], metrics_dict, resources_path)
        output_path = Path(ofolder_path) / f'report_{group}.pdf'
        with PdfPages(str(output_path)) as pdf:
            _pdf_add_cover_page(
                pdf=pdf,
                page_size=page_size,
                subject_name=Path(ofolder_path).name,
                group=group,
                subject_img=str(imgs_path / 'raw_and_seg_overlay.png'),
            )

            struc = 'canal'
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 1
            scale = _font_scale_for_grid(page_size, nrows=nrows, ncols=ncols)
            header_fs = _fs(45, scale, min_fs=12, max_fs=90)
            tick_fs = _fs(25, scale, min_fs=8, max_fs=60)
            suptitle_fs = _fs(120, scale, min_fs=14, max_fs=150)
            canal_tick_fs = _fs(18, scale, min_fs=10, max_fs=80)
            fig, axes = plt.subplots(nrows, ncols, figsize=page_size)
            axes = axes.flatten()
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 1]}.jpg')):
                        # Load image 
                        img_path = os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 1]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i-1], fontsize=header_fs, ha='center', va='center', fontweight='bold')
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=header_fs, ha='center', va='center')
                axes[idx].set_axis_off()
                idx += 1
                for metric in metrics:
                    ax = axes[idx]
                    if metric in subject_data[struc][struc_name]:
                        y_subject = subject_data[struc][struc_name][metric]
                        x_subject = subject_data[struc][struc_name]['slice_interp']

                        # Keep lines with metrics line equal to metric
                        all_values_data = all_values_df[group][struc][struc_name][metric]
                        
                        # Use seaborn line plot
                        sns.lineplot(x='slice_interp', y='values', data=all_values_data, ax=ax, errorbar='sd', color='gray')

                        # Plot subject
                        ax.plot(x_subject, y_subject, color='red', linewidth=2)

                        # Larger/more readable ticks for canal
                        tick_len = float(np.clip(6.0 * scale, 3.0, 12.0))
                        tick_w = float(np.clip(1.2 * scale, 0.8, 2.5))
                        ax.tick_params(axis='both', labelsize=canal_tick_fs, length=tick_len, width=tick_w, pad=2)
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
                        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                        
                        # Add vertebrae labels
                        disc = last_disc
                        top_pos = 0
                        nb_discs = all_values_data['slice_interp'].max()//discs_gap
                        # Leave some headroom for vertebra labels
                        ax.margins(y=0.18)
                        vertebra_label_fs = _fs(16, scale, min_fs=8, max_fs=42)
                        dense_labels = nb_discs >= 12
                        label_rot = 35 if dense_labels else 0
                        label_ha = 'right' if dense_labels else 'center'
                        for i in range(nb_discs+1):
                            top_vert = disc.split('-')[0]
                            ax.axvline(x=top_pos, color='gray', linestyle='--', alpha=0.5)
                            ax.text(
                                top_pos + discs_gap // 2,
                                ax.get_ylim()[1],
                                top_vert,
                                verticalalignment='bottom',
                                horizontalalignment=label_ha,
                                rotation=label_rot,
                                fontsize=vertebra_label_fs,
                                color='black',
                                alpha=0.7,
                            )
                            top_pos += discs_gap
                            if disc != 'C1-C2':
                                disc = previous_structure(disc)

                        ax.set_xlabel('')
                    else:
                        ax.set_axis_off()
                    idx += 1

            fig.suptitle(structure_titles.get(struc, struc), fontsize=suptitle_fs, fontweight='bold', y=0.985)
            _apply_report_grid_layout(fig, scale=scale, rotated_xticks=False)
            pdf.savefig(fig)
            _save_individual_figure(fig, images_dir, f"compared_{group}_{struc}")
            plt.close(fig)

            # Create vertebrae, foramens figures
            struc = 'foramens'
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 2
            scale = _font_scale_for_grid(page_size, nrows=nrows, ncols=ncols)
            header_fs = _fs(45, scale, min_fs=12, max_fs=90)
            tick_fs = _fs(25, scale, min_fs=8, max_fs=60)
            value_fs = _fs(25, scale, min_fs=8, max_fs=60)
            suptitle_fs = _fs(120, scale, min_fs=14, max_fs=150)
            fig = plt.figure(figsize=page_size)
            axes = _axes_with_big_header_row(
                fig,
                nrows=nrows,
                ncols=ncols,
                header_height=1.8,
                header_wspace=0.04,
                body_wspace=0.28,
                body_hspace=0.65,
                outer_hspace=0.04,
            )
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 2]}.jpg')):
                        # Load image 
                        img_path = os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 2]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i - 2], fontsize=header_fs, ha='center', va='center', fontweight='bold')
                
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=header_fs, ha='center', va='center')
                axes[idx].set_axis_off()
                img_name = f'{struc_name}'
                img_left = plt.imread(str(imgs_path / f'{img_name}_left.png'))
                img_right = plt.imread(str(imgs_path / f'{img_name}_right.png'))

                # Concatenate images after padding to the maximal shape
                max_height = max(img_left.shape[0], img_right.shape[0])
                img_left_padded = np.pad(np.fliplr(img_left), ((0, max_height - img_left.shape[0]), (0, 0)), mode='constant')
                img_right_padded = np.pad(img_right, ((0, max_height - img_right.shape[0]), (0, 0)), mode='constant')
                img = np.concatenate((img_right_padded, img_left_padded), axis=1)

                axes[idx+1].imshow(img)
                axes[idx+1].set_axis_off()
                idx += 2
                for metric in metrics:
                    ax = axes[idx]
                    add_group = False
                    add_subject = False
                    if metric in all_values_df[group][struc][struc_name]:
                        all_values_data = all_values_df[group][struc][struc_name][metric]
                        add_group = True
                    
                    if metric in subject_data[struc][struc_name]:
                        subject_value = subject_data[struc][struc_name][metric]
                        if subject_value != -1:
                            add_subject = True

                    # Plot metrics
                    if add_group:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                        ax.set_yticks([])
                    if add_subject:
                        # Overlay subject value (red) on top of the distribution
                        ax.axvline(x=subject_value, color='red', linestyle='--', linewidth=2, zorder=10)
                        ax.scatter([subject_value], [0], color='red', s=60, zorder=11)
                        try:
                            subject_value_float = float(subject_value)
                            if not np.isnan(subject_value_float):
                                ax.text(
                                    0.98,
                                    0.92,
                                    f"{subject_value_float:.3g}",
                                    transform=ax.transAxes,
                                    ha='right',
                                    va='top',
                                    color='red',
                                    fontsize=value_fs,
                                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', alpha=0.85),
                                    zorder=25,
                                )
                        except Exception:
                            pass
                        if not add_group:
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_yticks([])
                            span = max(1e-6, abs(float(subject_value)) * 0.2)
                            ax.set_xlim(float(subject_value) - span, float(subject_value) + span)

                    if add_group or add_subject:
                        ax.tick_params(axis='x', rotation=45, labelsize=tick_fs)
                        for lbl in ax.get_xticklabels():
                            lbl.set_horizontalalignment('right')
                        ax.set_xlabel('')
                    if not add_group and not add_subject:
                        ax.set_axis_off()
                    
                    idx += 1

            fig.suptitle(structure_titles.get(struc, struc), fontsize=suptitle_fs, fontweight='bold', y=0.985)
            _apply_report_outer_margins(fig, scale=scale, rotated_xticks=True)
            pdf.savefig(fig)
            _save_individual_figure(fig, images_dir, f"compared_{group}_{struc}")
            plt.close(fig)

            # Create discs figures
            struc = 'discs'
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 4
            scale = _font_scale_for_grid(page_size, nrows=nrows, ncols=ncols)
            header_fs = _fs(45, scale, min_fs=12, max_fs=100)
            tick_fs = _fs(25, scale, min_fs=8, max_fs=70)
            value_fs = _fs(25, scale, min_fs=8, max_fs=70)
            suptitle_fs = _fs(120, scale, min_fs=14, max_fs=150)
            fig = plt.figure(figsize=page_size)
            axes = _axes_with_big_header_row(
                fig,
                nrows=nrows,
                ncols=ncols,
                header_height=1.8,
                header_wspace=0.04,
                body_wspace=0.28,
                body_hspace=0.65,
                outer_hspace=0.04,
            )
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Disc grading", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                elif i == 2:
                    axes[i].text(0.5, 0.5, "Image", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                elif i == 3:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 4]}.jpg')):
                        # Load image 
                        img_path = os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 4]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i - 4], fontsize=header_fs, ha='center', va='center', fontweight='bold')
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=header_fs, ha='center', va='center')
                axes[idx].set_axis_off()
                grading = subject_data[struc][struc_name]['grading'][group]
                axes[idx+1].text(0.5, 0.5, f'Grading {grading}', fontsize=header_fs, ha='center', va='center')
                axes[idx+1].set_axis_off()
                # Load images
                img_name = f'{struc}_{struc_name}'
                img = plt.imread(str(imgs_path / f'{img_name}_img.png'))
                seg = plt.imread(str(imgs_path / f'{img_name}_seg.png'))
                axes[idx+2].imshow(np.rot90(img), cmap='gray')
                axes[idx+2].set_axis_off()
                axes[idx+3].imshow(np.rot90(seg))
                axes[idx+3].set_axis_off()
                idx += 4
                for metric in metrics:
                    ax = axes[idx]
                    add_group = False
                    add_subject = False
                    if metric in all_values_df[group][struc][struc_name]:
                        all_values_data = all_values_df[group][struc][struc_name][metric]
                        add_group = True
                    
                    if metric in subject_data[struc][struc_name]:
                        subject_value = subject_data[struc][struc_name][metric]
                        if subject_value != -1:
                            add_subject = True

                    # Plot metrics
                    if add_group:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                        ax.set_yticks([])
                    if add_subject:
                        ax.axvline(x=subject_value, color='red', linestyle='--', linewidth=2, zorder=10)
                        ax.scatter([subject_value], [0], color='red', s=60, zorder=11)
                        try:
                            subject_value_float = float(subject_value)
                            if not np.isnan(subject_value_float):
                                ax.text(
                                    0.98,
                                    0.92,
                                    f"{subject_value_float:.3g}",
                                    transform=ax.transAxes,
                                    ha='right',
                                    va='top',
                                    color='red',
                                    fontsize=value_fs,
                                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', alpha=0.85),
                                    zorder=35,
                                )
                        except Exception:
                            pass
                        if not add_group:
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_yticks([])
                            span = max(1e-6, abs(float(subject_value)) * 0.2)
                            ax.set_xlim(float(subject_value) - span, float(subject_value) + span)

                    if add_group or add_subject:
                        ax.tick_params(axis='x', rotation=45, labelsize=tick_fs)
                        for lbl in ax.get_xticklabels():
                            lbl.set_horizontalalignment('right')
                        ax.set_xlabel('')
                    if not add_group and not add_subject:
                        ax.set_axis_off()
                    
                    idx += 1

            fig.suptitle(structure_titles.get(struc, struc), fontsize=suptitle_fs, fontweight='bold', y=0.985)
            _apply_report_outer_margins(fig, scale=scale, rotated_xticks=True)
            pdf.savefig(fig)
            _save_individual_figure(fig, images_dir, f"compared_{group}_{struc}")
            plt.close(fig)
        
            struc = 'vertebrae'
            # Create a subplot for each subject and overlay a red line corresponding to their value
            struc_names = np.array(list(subject_data[struc].keys()))
            struc_names = struc_names[np.isin(struc_names, list(all_values_df[group][struc].keys()))].tolist()
            metrics = metrics_dict[struc]
            nrows = len(struc_names) + 1
            ncols = len(metrics) + 3
            scale = _font_scale_for_grid(page_size, nrows=nrows, ncols=ncols)
            header_fs = _fs(45, scale, min_fs=12, max_fs=90)
            tick_fs = _fs(25, scale, min_fs=8, max_fs=60)
            value_fs = _fs(25, scale, min_fs=8, max_fs=60)
            suptitle_fs = _fs(120, scale, min_fs=14, max_fs=150)
            fig = plt.figure(figsize=page_size)
            axes = _axes_with_big_header_row(
                fig,
                nrows=nrows,
                ncols=ncols,
                header_height=1.8,
                header_wspace=0.04,
                body_wspace=0.28,
                body_hspace=0.65,
                outer_hspace=0.04,
            )
            idx = 0
            for i in range(ncols):
                if i == 0:
                    axes[i].text(0.5, 0.5, "Structure name", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                elif i == 1:
                    axes[i].text(0.5, 0.5, "Image", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                elif i == 2:
                    axes[i].text(0.5, 0.5, "Segmentation", fontsize=header_fs, ha='center', va='center', fontweight='bold')
                else:
                    if os.path.exists(os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 3]}.jpg')):
                        # Load image 
                        img_path = os.path.join(resources_path, f'imgs/{struc}_{metrics[i - 3]}.jpg')
                        axes[i].imshow(plt.imread(img_path))
                    else:
                        axes[i].text(0.5, 0.5, metrics[i - 3], fontsize=header_fs, ha='center', va='center', fontweight='bold')
                axes[i].set_axis_off()
                idx += 1
            for struc_name in struc_names:
                axes[idx].text(0.5, 0.5, struc_name, fontsize=header_fs, ha='center', va='center')
                axes[idx].set_axis_off()
                # Load images
                img_name = f'{struc}_{struc_name}'
                img = plt.imread(str(imgs_path / f'{img_name}_img.png'))
                seg = plt.imread(str(imgs_path / f'{img_name}_seg.png'))
                axes[idx+1].imshow(np.rot90(img), cmap='gray')
                axes[idx+1].set_axis_off()
                axes[idx+2].imshow(np.rot90(seg))
                axes[idx+2].set_axis_off()
                idx += 3
                for metric in metrics:
                    ax = axes[idx]
                    add_group = False
                    add_subject = False
                    if metric in all_values_df[group][struc][struc_name]:
                        all_values_data = all_values_df[group][struc][struc_name][metric]
                        add_group = True
                    
                    if metric in subject_data[struc][struc_name]:
                        subject_value = subject_data[struc][struc_name][metric]
                        if subject_value != -1:
                            add_subject = True

                    # Plot metrics
                    if add_group:
                        sns.violinplot(x='values', data=all_values_data, ax=ax, cut=0, bw_method=0.7, color='gray', alpha=0.2)
                        ax.set_yticks([])
                    if add_subject:
                        ax.axvline(x=subject_value, color='red', linestyle='--', linewidth=2, zorder=10)
                        ax.scatter([subject_value], [0], color='red', s=60, zorder=11)
                        try:
                            subject_value_float = float(subject_value)
                            if not np.isnan(subject_value_float):
                                ax.text(
                                    0.98,
                                    0.92,
                                    f"{subject_value_float:.3g}",
                                    transform=ax.transAxes,
                                    ha='right',
                                    va='top',
                                    color='red',
                                    fontsize=value_fs,
                                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', alpha=0.85),
                                    zorder=25,
                                )
                        except Exception:
                            pass
                        if not add_group:
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_yticks([])
                            span = max(1e-6, abs(float(subject_value)) * 0.2)
                            ax.set_xlim(float(subject_value) - span, float(subject_value) + span)

                    if add_group or add_subject:
                        ax.tick_params(axis='x', rotation=45, labelsize=tick_fs)
                        for lbl in ax.get_xticklabels():
                            lbl.set_horizontalalignment('right')
                        ax.set_xlabel('')
                    if not add_group and not add_subject:
                        ax.set_axis_off()
                    
                    idx += 1

            fig.suptitle(structure_titles.get(struc, struc), fontsize=suptitle_fs, fontweight='bold', y=0.985)
            _apply_report_outer_margins(fig, scale=scale, rotated_xticks=True)
            pdf.savefig(fig)
            _save_individual_figure(fig, images_dir, f"compared_{group}_{struc}")
            plt.close(fig)

def generate_pdf(subject_name, group, subject_img, figures_path, out_path):
    raise NotImplementedError(
        'PDF generation is now handled directly inside create_global_figures() to avoid rasterizing figures to PNG first.'
    )


def convert_str_to_list(string):
    return [float(item.strip()) for item in string[1:-1].split(',')]


def categorize_age_groups(age):
    """
    Categorize age into groups.
    
    Args:
        age: Age value or list of ages
    
    Returns:
        Age group label
    """
    if age < 40:
        return '18-39'
    elif age < 60:
        return '40-59'
    else:
        return '60+'

if __name__ == "__main__":
    test_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/metrics_output' # '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/test-tss/out/metrics_output'
    control_path = '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/datasets/analysis_balgrist/out/metrics_output'
    ofolder = 'test'
    quiet = False
    generate_reports(
        test_path=test_path,
        control_path=control_path,
        ofolder_path=ofolder,
        max_workers=8,
        quiet=quiet,
    )