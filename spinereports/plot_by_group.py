import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import copy
from pathlib import Path
import json
import argparse, textwrap
import importlib
import multiprocessing as mp

import totalspineseg.resources as resources
from spinereports.utils.measure_seg import measure_seg_mp

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script generates group plots from segmentation measurements.
            It requires files to follow the BIDS naming conventions. 
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            spinereports_plot_by_group -i group_folder -d demographics.tsv -o reports
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input-dir', '-i', type=Path, required=True,
        help='The folder where totalspineseg outputs are located (required).'
    )
    parser.add_argument(
        '--demographics', '-d', type=Path, required=True,
        help='The demographics file (required). With a column corresponding to the "participant_id", the "sex" and "age" columns.'
    )
    parser.add_argument(
        '--ofolder', '-o', type=Path, required=True,
        help='The folder where reports will be saved (required).'
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
    folder_path = args.input_dir
    demographics_path = args.demographics
    output_folder = args.ofolder
    prefix = args.prefix
    image_suffix = args.image_suffix
    seg_suffix = args.seg_suffix
    label_suffix = args.label_suffix
    max_workers = args.max_workers
    quiet = args.quiet

    # Use default mapping path
    resources_path = importlib.resources.files(resources)
    seg_mapping_path = resources_path / 'labels_maps/tss_map.json'

    # Check if paths exists
    if not folder_path.exists():
        raise FileNotFoundError(f'Directory "{folder_path}" does not exist.')

    # Measure segmentations if not already done
    folder_metrics_path = folder_path / "metrics_output"
    if not folder_metrics_path.exists():
        if not quiet: print(f'\nMeasuring segmentations for group in "{folder_path}"...')
        measure_seg_mp(
            images_path=folder_path / "input",
            segs_path=folder_path / "step2_output",
            labels_path=folder_path / "step1_levels",
            ofolder_path=folder_metrics_path,
            prefix=prefix,
            image_suffix=image_suffix,
            seg_suffix=seg_suffix,
            label_suffix=label_suffix,
            mapping_path=seg_mapping_path,
            max_workers=max_workers,
            quiet=quiet,
        )
        
    # Generate group plots
    if not quiet: print(f'\nGenerating group plots for folder "{folder_metrics_path}"...')
    demographics = pd.read_csv(demographics_path, sep='\t')
    all_values = {}
    all_demographics = {}
    for sub in os.listdir(folder_metrics_path):
        if sub.startswith('sub-'):
            sub_folder = folder_metrics_path / sub
            sub_name = sub.split('_')[0]
            sub_info = demographics[demographics['participant_id'] == sub_name]
            sub_info = df_to_dict(sub_info)

            # Compute metrics subject
            control_data = compute_metrics_subject(sub_folder)

            # Gather all values for each metric and structures
            for struc in control_data.keys():
                for struc_name in control_data[struc].keys():
                    if struc_name != 'spinalcord/canal':
                        for metric in control_data[struc][struc_name].keys():
                            # Add subject to all_values
                            subject_value = control_data[struc][struc_name][metric]
                            if subject_value != -1 and isinstance(sub_info['age'], (int, float)) and sub_info['sex'] in ['M', 'F']:
                                if struc not in all_values:
                                    all_values[struc] = {}
                                    all_demographics[struc] = {}
                                if struc_name not in all_values[struc]:
                                    all_values[struc][struc_name] = {}
                                    all_demographics[struc][struc_name] = {}
                                if metric not in all_values[struc][struc_name]:
                                    all_values[struc][struc_name][metric] = []
                                    all_demographics[struc][struc_name][metric] = []
                                all_values[struc][struc_name][metric].append(subject_value)
                                all_demographics[struc][struc_name][metric].append(sub_info)
        
    # Align canal and CSF for control group
    levels_mapping_path = resources_path / 'labels_maps/levels_maps.json'
    with open(str(levels_mapping_path), 'r') as f:
        mapping = json.load(f)
        rev_mapping = {v: k for k, v in mapping.items()}
    all_values, discs_gap, last_disc = rescale_canal(all_values, rev_mapping)

    # OVERALL ROBUSTNESS ANALYSIS (not grouped by demographics)
    print("Starting overall robustness analysis...")
    generate_robustness_summary_overall(all_values, all_demographics, output_folder)

    # GROUP ANALYSIS AND ROBUSTNESS EVALUATION
    print("Starting demographic group analysis and robustness evaluation...")
    
    # Group data by demographics separately for sex and age
    grouped_by_sex, grouped_by_age = group_data_by_demographics(all_values, all_demographics)

    all_values_df = group_canal_by_demographics(all_values, all_demographics)
    
    # Create output folders
    output_folder.mkdir(exist_ok=True)
    sex_output_folder = output_folder / 'by_sex'
    age_output_folder = output_folder / 'by_age'
    sex_output_folder.mkdir(exist_ok=True)
    age_output_folder.mkdir(exist_ok=True)
    
    # Generate plots by sex
    print("Generating plots by sex...")
    plot_metrics_by_sex(grouped_by_sex, sex_output_folder)
    
    # Generate plots by age
    print("Generating plots by age...")
    plot_metrics_by_age(grouped_by_age, age_output_folder)
    
    # Generate canal plots by sex
    print("Generating canal plots by sex...")
    plot_canal_metrics_by_sex(all_values_df, discs_gap, last_disc, sex_output_folder)
    
    # Generate canal plots by age
    print("Generating canal plots by age...")
    plot_canal_metrics_by_age(all_values_df, discs_gap, last_disc, age_output_folder)
    
    print(f"Analysis complete! Results saved to {output_folder}")

def sort_anatomical_structures(structure_list):
    """
    Sort anatomical structures in correct anatomical order.
    
    Args:
        structure_list: List of structure names to sort
        
    Returns:
        List of structures sorted in anatomical order
    """
    def get_sort_key(structure):
        """Generate sorting key for anatomical structures."""
        structure = structure.strip()
        
        # Handle vertebrae (T9, T10, T11, T12, L1, L2, L3, L4, L5, S1)
        if re.match(r'^T\d+$', structure):
            return (0, int(structure[1:]))  # Thoracic vertebrae
        elif re.match(r'^L\d+$', structure):
            return (1, int(structure[1:]))  # Lumbar vertebrae
        elif re.match(r'^S\d+$', structure):
            return (2, int(structure[1:]))  # Sacral vertebrae
        
        # Handle discs (T9-T10, T10-T11, ..., T12-L1, L1-L2, ..., L5-S1)
        elif '-' in structure:
            parts = structure.split('-')
            if len(parts) == 2:
                upper, lower = parts[0].strip(), parts[1].strip()
                
                # Get numeric values for both parts
                upper_key = get_sort_key(upper)
                lower_key = get_sort_key(lower)
                
                # Use the upper vertebra as primary sort key for discs
                return upper_key
        
        # Handle foramens (foramens_T11-T12, foramens_T12-L1, foramens_L1-L2)
        elif structure.startswith('foramens_'):
            disc_part = structure.replace('foramens_', '')
            return get_sort_key(disc_part)
        
        # Default case - try to extract any vertebral level information
        else:
            # Look for T/L/S patterns in the string
            match = re.search(r'([TLS])(\d+)', structure)
            if match:
                letter, number = match.groups()
                if letter == 'T':
                    return (0, int(number))
                elif letter == 'L':
                    return (1, int(number))
                elif letter == 'S':
                    return (2, int(number))
        
        # If no pattern matches, sort alphabetically at the end
        return (9999, structure)
    
    return sorted(structure_list, key=get_sort_key)

def is_vertebra_in_range(vertebra, min_level='T11', max_level='L5'):
    """
    Check if a vertebra is within the specified anatomical range.
    
    Args:
        vertebra: Vertebra name (e.g., 'T11', 'L3')
        min_level: Minimum vertebra level (default 'T11')
        max_level: Maximum vertebra level (default 'L5')
    
    Returns:
        Boolean indicating if vertebra is in range
    """
    def vertebra_to_numeric(vert):
        """Convert vertebra to numeric for comparison."""
        if re.match(r'^T\d+$', vert):
            return (0, int(vert[1:]))  # Thoracic: (0, number)
        elif re.match(r'^L\d+$', vert):
            return (1, int(vert[1:]))  # Lumbar: (1, number)
        elif re.match(r'^S\d+$', vert):
            return (2, int(vert[1:]))  # Sacral: (2, number)
        return (999, 0)  # Unknown
    
    vert_num = vertebra_to_numeric(vertebra)
    min_num = vertebra_to_numeric(min_level)
    max_num = vertebra_to_numeric(max_level)
    
    return min_num <= vert_num <= max_num

def is_disc_in_range(disc, min_level='T10-T11', max_level='L5-S1'):
    """
    Check if a disc is within the specified anatomical range.
    
    Args:
        disc: Disc name (e.g., 'T10-T11', 'L4-L5')
        min_level: Minimum disc level (default 'T10-T11')
        max_level: Maximum disc level (default 'L5-S1')
    
    Returns:
        Boolean indicating if disc is in range
    """
    if '-' not in disc:
        return False
    
    # Extract upper vertebra from disc
    upper_vert = disc.split('-')[0].strip()
    
    # For discs, use the upper vertebra to determine position
    min_upper = min_level.split('-')[0].strip()
    max_upper = max_level.split('-')[0].strip()
    
    return is_vertebra_in_range(upper_vert, min_upper, max_upper)

def is_foramen_in_range(foramen, min_level='T10-T11', max_level='L5-S1'):
    """
    Check if a foramen is within the specified anatomical range.
    
    Args:
        foramen: Foramen name (e.g., 'foramens_T10-T11', 'foramens_L4-L5')
        min_level: Minimum disc level (default 'T10-T11')
        max_level: Maximum disc level (default 'L5-S1')
    
    Returns:
        Boolean indicating if foramen is in range
    """
    if not foramen.startswith('foramens_'):
        return False
    
    # Extract disc part from foramen
    disc_part = foramen.replace('foramens_', '')
    return is_disc_in_range(disc_part, min_level, max_level)

def filter_structures_by_range(structure_list, structure_type='general', min_level=None, max_level=None):
    """
    Filter structures based on anatomical range.
    
    Args:
        structure_list: List of structure names
        structure_type: Type of structures ('vertebrae', 'discs', 'foramens', 'canal', 'general')
        min_level: Minimum anatomical level
        max_level: Maximum anatomical level
    
    Returns:
        Filtered list of structures
    """
    if min_level is None or max_level is None:
        return structure_list
    
    filtered_structures = []
    
    for structure in structure_list:
        include_structure = False
        
        if structure_type == 'vertebrae':
            include_structure = is_vertebra_in_range(structure, min_level, max_level)
        elif structure_type == 'discs':
            include_structure = is_disc_in_range(structure, min_level, max_level)
        elif structure_type == 'foramens':
            include_structure = is_foramen_in_range(structure, min_level, max_level)
        elif structure_type == 'canal':
            # For canal, we filter based on vertebrae range
            include_structure = is_vertebra_in_range(structure, min_level, max_level) if re.match(r'^[TLS]\d+$', structure) else True
        else:  # general - try to determine type automatically
            if re.match(r'^[TLS]\d+$', structure):
                include_structure = is_vertebra_in_range(structure, min_level, max_level)
            elif '-' in structure and not structure.startswith('foramens_'):
                include_structure = is_disc_in_range(structure, min_level, max_level)
            elif structure.startswith('foramens_'):
                include_structure = is_foramen_in_range(structure, min_level, max_level)
            else:
                include_structure = True  # Include unknown structures
        
        if include_structure:
            filtered_structures.append(structure)
    
    return filtered_structures


def group_canal_by_demographics(all_values, all_demographics):
    new_values = copy.deepcopy(all_values)
    for struc in ['canal', 'csf']:
        for struc_name in new_values[struc].keys():
            # Convert dict to dataframe with keys as columns and lines as subjects
            # Prepare a dictionary where each key is a metric and each value is a list of values for all subjects
            for i, metric in enumerate(all_values[struc][struc_name].keys()):
                if metric not in ['discs_gap', 'slice_interp']:
                    data = {'subjects' : [], 'values' : [], 'slice_interp' : [], 'sex': [], 'age': []}
                    for j, subject_value in enumerate(all_values[struc][struc_name][metric]):
                        if isinstance(subject_value, list):
                            for value, slice_interp in zip(subject_value, all_values[struc][struc_name]['slice_interp'][j]):
                                data['values'].append(value)
                                data['slice_interp'].append(slice_interp)
                                data['subjects'].append(f'subject_{j}')
                                if all_demographics[struc][struc_name][metric][j]['sex'] == 'M':
                                    data['sex'].append('Male')
                                elif all_demographics[struc][struc_name][metric][j]['sex'] == 'F':
                                    data['sex'].append('Female')
                                else:
                                    raise ValueError("Unexpected sex value encountered.")
                                age_group = categorize_age_groups(all_demographics[struc][struc_name][metric][j]['age'])
                                data['age'].append(age_group)
                        else:
                            raise ValueError("Expected list of values for canal/csf metrics.")
                    # Reverse slice_interp to go from top to bottom
                    max_val = max(data['slice_interp'])
                    data['slice_interp'] = [max_val-val for val in data['slice_interp']]
                
                df = pd.DataFrame.from_dict(
                    data,
                    orient='index'
                ).transpose()
                new_values[struc][struc_name][metric] = df
    return new_values

def df_to_dict(df):
    idx = df['participant_id'].keys()[0]
    d = {k:v[idx] for k,v in df.to_dict().items()}
    return d

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
    subject_dict = create_dict_from_subject_data(subject_data, intensity_profile=False)
    return subject_dict

def process_foramens(subject_data):
    # Create dictionary from pandas dataframes with names as keys
    subject_dict = create_dict_from_subject_data(subject_data)
    return subject_dict

def compute_discs_metrics(data_dict):
    # Compute Disc Height Index (DHI)
    for struc_name in data_dict['discs'].keys():
        top_vertebra = struc_name.split('-')[0]
        if top_vertebra in data_dict['vertebrae']:
            # Normalize disc height with top vertebra AP_thickness
            data_dict['discs'][struc_name]['DHI'] = data_dict['discs'][struc_name]['median_thickness'] / data_dict['vertebrae'][top_vertebra]['AP_thickness']
        else:
            data_dict['discs'][struc_name]['DHI'] = -1
    return data_dict

def compute_foramens_metrics(data_dict):
    # Compute Foramen metrics
    for struc_name in data_dict['foramens'].keys():
        top_vertebra = struc_name.replace('foramens_','').split('-')[0]
        if not top_vertebra in data_dict['vertebrae']:
            data_dict['foramens'][struc_name]['right_surface'] = -1
            data_dict['foramens'][struc_name]['left_surface'] = -1
            data_dict['foramens'][struc_name]['asymmetry_R-L'] = -1
        else:
            # Normalize foramen surfaces with top vertebra AP thickness
            for surface in ['right_surface', 'left_surface']:
                if data_dict['foramens'][struc_name][surface] != -1:
                    data_dict['foramens'][struc_name][surface] = data_dict['foramens'][struc_name][surface] / (data_dict['vertebrae'][top_vertebra]['AP_thickness']*data_dict['vertebrae'][top_vertebra]['median_thickness'])

            # Create asymmetry quotient
            if data_dict['foramens'][struc_name]['right_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != -1 and data_dict['foramens'][struc_name]['left_surface'] != 0:
                data_dict['foramens'][struc_name]['asymmetry_R-L'] = data_dict['foramens'][struc_name]['right_surface'] / data_dict['foramens'][struc_name]['left_surface']
            else:
                data_dict['foramens'][struc_name]['asymmetry_R-L'] = -1
    return data_dict

def compute_vertebrae_metrics(data_dict):
    # Compute Vertebrae metrics
    # for struc_name in data_dict['vertebrae'].keys():
    #     # Normalize foramen surfaces with top vertebra volume
    #     for metric in data_dict['vertebrae'][struc_name].keys():
    #         if metric != 'volume':
    #             data_dict['vertebrae'][struc_name][metric] = data_dict['vertebrae'][struc_name][metric] / data_dict['vertebrae'][struc_name]['volume']
    return data_dict

def rescale_canal(all_values, rev_mapping):
    '''
    Rescale subject canals and CSF based on discs z coordinates.
    '''
    new_values = copy.deepcopy(all_values)
    struc = 'canal'
    struc_name = 'canal'
    # Align all metrics for each subject using discs level as references
    disc_levels = all_values[struc][struc_name]['disc_level']
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

    for struc in ['canal', 'csf']:
        for struc_name in all_values[struc].keys():
            # Rescale subjects
            add_slice_interp = True
            for metric in all_values[struc][struc_name].keys():
                if metric in ['slice_nb', 'disc_level']:
                    continue
                for subj_idx in range(len(all_values[struc][struc_name][metric])):
                    interp_values, slice_interp = rescale_with_discs(all_values[struc][struc_name]['disc_level'][subj_idx], all_values[struc][struc_name][metric][subj_idx], rev_mapping, discs_gap, last_disc)
                    new_values[struc][struc_name][metric][subj_idx] = interp_values
                    if 'slice_interp' not in new_values[struc][struc_name]:
                        new_values[struc][struc_name]['slice_interp'] = []
                    if add_slice_interp:
                        new_values[struc][struc_name]['slice_interp'].append(slice_interp)
                add_slice_interp = False
            # Remove slice_nb and disc_level from dict
            new_values[struc][struc_name].pop('slice_nb', None)
            new_values[struc][struc_name].pop('disc_level', None)

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
    while disc != rev_mapping[int(subj_disc_values[0])]:
        start_disc_gap += gap
        disc = previous_structure(disc)
    slice_interp += list(range(start_disc_gap, start_disc_gap + len(interp_values)))
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

def create_dict_from_subject_data(subject_data, intensity_profile=True):
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

def calculate_robustness_metrics_overall(all_values, all_demographics):
    """
    Calculate robustness metrics for subjects with multiple measurements across all data.
    
    Args:
        all_values: Dictionary with structure {struc: {struc_name: {metric: [values]}}}
        all_demographics: Dictionary with structure {struc: {struc_name: {metric: [demographics]}}}
        
    Returns:
        DataFrame with robustness metrics per subject, structure, and metric
    """
    robustness_data = []
    
    for struc in ['discs', 'foramens', 'vertebrae']:
        for struc_name in all_values[struc].keys():
            for metric in all_values[struc][struc_name].keys():
                if metric in ['discs_gap', 'slice_interp']:  # Skip non-metric data
                    continue
                    
                values = all_values[struc][struc_name][metric]
                demographics = all_demographics[struc][struc_name][metric]
                
                # Group by subject (participant_id)
                subject_values = {}
                for val, demo in zip(values, demographics):
                    subj_id = demo['participant_id']
                    if subj_id not in subject_values:
                        subject_values[subj_id] = []
                    subject_values[subj_id].append(val)
                
                # Calculate robustness for subjects with multiple measurements
                for subj_id, subj_vals in subject_values.items():
                    if len(subj_vals) > 1:  # Multiple measurements
                        subj_vals = np.array(subj_vals)
                        subj_vals = subj_vals[~np.isnan(subj_vals)]  # Remove NaN
                        
                        if len(subj_vals) > 1:
                            mean_val = np.mean(subj_vals)
                            std_val = np.std(subj_vals)
                            cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
                            min_val = np.min(subj_vals)
                            max_val = np.max(subj_vals)
                            range_val = max_val - min_val
                            relative_range = (range_val / mean_val) * 100 if mean_val != 0 else np.nan
                            
                            robustness_data.append({
                                'structure': struc,
                                'structure_name': struc_name,
                                'metric': metric,
                                'subject': subj_id,
                                'n_measurements': len(subj_vals),
                                'mean': mean_val,
                                'std': std_val,
                                'cv_percent': cv,
                                'min': min_val,
                                'max': max_val,
                                'range': range_val,
                                'relative_range_percent': relative_range,
                                'measurements': subj_vals.tolist()
                            })
    
    for struc in ['canal', 'csf']:
        for struc_name in all_values[struc].keys():
            for metric in all_values[struc][struc_name].keys():
                if metric in ['discs_gap', 'slice_interp']:  # Skip non-metric data
                    continue
                    
                values = all_values[struc][struc_name][metric]
                slice_interp = all_values[struc][struc_name]['slice_interp']
                demographics = all_demographics[struc][struc_name][metric]
                
                # Group by subject (participant_id)
                subject_values = {}
                for sl, val, demo in zip(slice_interp, values, demographics):
                    subj_id = demo['participant_id']
                    if subj_id not in subject_values:
                        subject_values[subj_id] = {'slice_interp': [], 'values': []}
                    subject_values[subj_id]['slice_interp'].append(sl)
                    subject_values[subj_id]['values'].append(val)
                
                # Calculate robustness for subjects with multiple measurements
                for subj_id, subj_data in subject_values.items():
                    if len(subj_data['values']) > 1:  # Multiple measurements
                        # Flatten the nested lists and create corresponding slice indices
                        all_vals = []
                        all_slices = []
                        
                        for session_vals, session_slices in zip(subj_data['values'], subj_data['slice_interp']):
                            # Handle case where session_vals might be a single value or a list
                            if isinstance(session_vals, list):
                                all_vals.extend(session_vals)
                                all_slices.extend(session_slices)
                            else:
                                all_vals.append(session_vals)
                                all_slices.append(session_slices)
                        
                        subj_vals = np.array(all_vals)
                        subj_slice = np.array(all_slices)
                        
                        if len(subj_vals) == 0:
                            continue
                        
                        min_slice = np.min(subj_slice)
                        max_slice = np.max(subj_slice)

                        mean_list = []
                        std_list = []
                        for sl in range(min_slice, max_slice + 1):
                            mask = subj_slice == sl
                            if np.sum(mask) > 1:  # More than one measurement at this slice
                                subj_val = subj_vals[mask]  
                                subj_val = subj_val[~np.isnan(subj_val)]  # Remove NaN

                                if len(subj_val) > 1:
                                    mean_list.append(np.mean(subj_val))
                                    std_list.append(np.std(subj_val))

                        mean_val = np.mean(mean_list) if mean_list else np.nan
                        std_val = np.mean(std_list) if std_list else np.nan
                        cv = (std_val / mean_val) * 100 if mean_val != 0 and not np.isnan(mean_val) else np.nan
                        min_val = np.min(mean_list) if mean_list else np.nan
                        max_val = np.max(mean_list) if mean_list else np.nan
                        range_val = max_val - min_val if mean_list else np.nan
                        relative_range = (range_val / mean_val) * 100 if mean_val != 0 and not np.isnan(mean_val) else np.nan
                                    
                        robustness_data.append({
                            'structure': struc,
                            'structure_name': struc_name,
                            'metric': metric,
                            'subject': subj_id,
                            'n_measurements': len(subj_vals),
                            'mean': mean_val,
                            'std': std_val,
                            'cv_percent': cv,
                            'min': min_val,
                            'max': max_val,
                            'range': range_val,
                            'relative_range_percent': relative_range,
                            'measurements': subj_vals.tolist()
                        })
    
    return pd.DataFrame(robustness_data)


def generate_robustness_summary_overall(all_values, all_demographics, output_folder):
    """
    Generate comprehensive summary statistics for robustness analysis (not grouped by demographics).
    
    Args:
        all_values: Dictionary with structure {struc: {struc_name: {metric: [values]}}}
        all_demographics: Dictionary with structure {struc: {struc_name: {metric: [demographics]}}}
        output_folder: Path to save summary files
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Calculate overall robustness metrics
    robustness_df = calculate_robustness_metrics_overall(all_values, all_demographics)
    
    if robustness_df.empty:
        print("No subjects with multiple measurements found for robustness analysis")
        return
    
    print(f"Found {len(robustness_df)} subjects with multiple measurements for robustness analysis")
    
    # Group by metric for summary statistics
    summary_data = []
    
    # Get all structures and sort them anatomically
    all_structures = robustness_df['structure'].unique()
    sorted_structures = sort_anatomical_structures(all_structures)
    
    for struc in sorted_structures:
        struc_data = robustness_df[robustness_df['structure'] == struc]
        for metric in struc_data['metric'].unique():
            metric_data = struc_data[struc_data['metric'] == metric]

            overall_stats = {
                'struc': struc,
                'metric': metric,
                'n_subjects_with_multiple': len(metric_data),
                'total_measurements': metric_data['n_measurements'].sum(),
                'mean_cv_percent': metric_data['cv_percent'].mean(),
                'median_cv_percent': metric_data['cv_percent'].median(),
                'std_cv_percent': metric_data['cv_percent'].std(),
                'mean_relative_range_percent': metric_data['relative_range_percent'].mean(),
                'median_relative_range_percent': metric_data['relative_range_percent'].median(),
                'subjects_with_high_variability_cv_20': (metric_data['cv_percent'] > 20).sum(),
                'subjects_with_low_variability_cv_5': (metric_data['cv_percent'] < 5).sum()
            }
            summary_data.append(overall_stats)
    
    # Create overall summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(3)
    # summary_df.to_csv(output_folder / 'overall_robustness_summary.csv', index=False)
    
    # Save detailed robustness data
    # robustness_df.to_csv(output_folder / 'detailed_robustness_data.csv', index=False)
    
    # Print summary to console
    print("\n" + "="*80)
    print("OVERALL ROBUSTNESS ANALYSIS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Create summary visualization
    plt.figure(figsize=(18, 12))
    
    # Plot 1: CV comparison across structures and metrics
    plt.subplot(2, 3, 1)
    # Create combined labels for structure-metric combinations
    summary_df['struc_metric'] = summary_df['struc'] + '_' + summary_df['metric']
    cv_means = summary_df['mean_cv_percent']
    bars = plt.bar(range(len(cv_means)), cv_means)
    plt.xticks(range(len(cv_means)), summary_df['struc_metric'], rotation=45, ha='right')
    plt.ylabel('Mean CV (%)')
    plt.title('Mean Coefficient of Variation by Structure-Metric')
    plt.grid(True, alpha=0.3)
    
    # Color bars by CV level
    for bar, cv in zip(bars, cv_means):
        if cv < 10:
            bar.set_color('green')
        elif cv < 20:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # Plot 2: Number of subjects with multiple measurements
    plt.subplot(2, 3, 2)
    plt.bar(range(len(summary_df)), summary_df['n_subjects_with_multiple'])
    plt.xticks(range(len(summary_df)), summary_df['struc_metric'], rotation=45, ha='right')
    plt.ylabel('Number of Subjects')
    plt.title('Subjects with Multiple Measurements')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: High vs Low variability subjects
    plt.subplot(2, 3, 3)
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width/2, summary_df['subjects_with_high_variability_cv_20'], 
            width, label='High variability (CV>20%)', alpha=0.7, color='red')
    plt.bar(x + width/2, summary_df['subjects_with_low_variability_cv_5'], 
            width, label='Low variability (CV<5%)', alpha=0.7, color='green')
    plt.xticks(x, summary_df['struc_metric'], rotation=45, ha='right')
    plt.ylabel('Number of Subjects')
    plt.title('Variability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: CV distribution by structure (boxplot)
    plt.subplot(2, 3, 4)
    structures = sort_anatomical_structures(robustness_df['structure'].unique())
    cv_data_by_structure = []
    structure_labels = []
    
    for structure in structures:
        struct_cv = robustness_df[robustness_df['structure'] == structure]['cv_percent'].dropna().values
        if len(struct_cv) > 0:
            cv_data_by_structure.append(struct_cv)
            structure_labels.append(structure)
    
    if cv_data_by_structure:
        plt.boxplot(cv_data_by_structure, labels=structure_labels)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('CV (%)')
        plt.title('CV Distribution by Structure')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Number of measurements distribution
    plt.subplot(2, 3, 5)
    n_meas_counts = robustness_df['n_measurements'].value_counts().sort_index()
    plt.bar(n_meas_counts.index, n_meas_counts.values)
    plt.xlabel('Number of Measurements')
    plt.ylabel('Number of Subjects')
    plt.title('Distribution of Measurement Counts')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: CV vs Number of measurements scatter
    plt.subplot(2, 3, 6)
    # Color by structure
    unique_structures = sort_anatomical_structures(robustness_df['structure'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_structures)))
    for i, structure in enumerate(unique_structures):
        struct_data = robustness_df[robustness_df['structure'] == structure]
        plt.scatter(struct_data['n_measurements'], struct_data['cv_percent'], 
                   alpha=0.6, label=structure, color=colors[i])
    plt.xlabel('Number of Measurements')
    plt.ylabel('CV (%)')
    plt.title('CV vs Number of Measurements')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_folder / 'overall_robustness_summary_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary by structure and metric
    structure_metric_summary = robustness_df.groupby(['structure', 'structure_name', 'metric']).agg({
        'cv_percent': ['count', 'mean', 'median', 'std'],
        'relative_range_percent': ['mean', 'median', 'std'],
        'n_measurements': 'mean'
    }).round(3)
    
    structure_metric_summary.columns = ['_'.join(col).strip() for col in structure_metric_summary.columns]
    #structure_metric_summary.to_csv(output_folder / 'robustness_by_structure_metric.csv')
    
    print(f"Overall robustness analysis saved to {output_folder}")
    print(f"Generated files:")
    print(f"  - Summary plots: overall_robustness_summary_plots.png")
    print(f"  - Summary table: overall_robustness_summary.csv")
    print(f"  - Detailed data: detailed_robustness_data.csv")
    print(f"  - By structure/metric: robustness_by_structure_metric.csv")
    


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


def group_data_by_demographics(all_values, all_demographics):
    """
    Group data by sex and age separately for analysis.
    
    Args:
        all_values: Dictionary with structure {struc: {struc_name: {metric: [values]}}}
        all_demographics: Dictionary with structure {struc: {struc_name: {metric: [demographics]}}}
    
    Returns:
        Two separate grouped data dictionaries: grouped_by_sex, grouped_by_age
    """
    grouped_by_sex = {}
    grouped_by_age = {}
    sex_dict = {'M': 'Male', 'F': 'Female'}
    
    for struc in ['discs', 'foramens', 'vertebrae']:
        grouped_by_sex[struc] = {}
        grouped_by_age[struc] = {}
        
        for struc_name in all_values[struc].keys():
            grouped_by_sex[struc][struc_name] = {}
            grouped_by_age[struc][struc_name] = {}
            
            for metric in all_values[struc][struc_name].keys():
                if metric in ['discs_gap', 'slice_interp']:  # Skip non-metric data
                    continue
                    
                values = all_values[struc][struc_name][metric]
                demographics = all_demographics[struc][struc_name][metric]
                slice_interp = all_values[struc][struc_name].get('slice_interp', None)
                
                # Group by sex
                sex_groups = {
                    'Male': {'values': [], 'demographics': [], 'slice_interp': []},
                    'Female': {'values': [], 'demographics': [], 'slice_interp': []}
                }
                
                # Group by age
                age_groups = {
                    '18-39': {'values': [], 'demographics': [], 'slice_interp': []},
                    '40-59': {'values': [], 'demographics': [], 'slice_interp': []},
                    '60+': {'values': [], 'demographics': [], 'slice_interp': []}
                }
                
                for idx, (val, demo) in enumerate(zip(values, demographics)):
                    age = demo['age']
                    sex = sex_dict[demo['sex']]
                    
                    # Add to sex groups
                    sex_groups[sex]['values'].append(val)
                    sex_groups[sex]['demographics'].append(demo)
                    if slice_interp is not None and idx < len(slice_interp):
                        sex_groups[sex]['slice_interp'].append(slice_interp[idx])

                    # Add to age groups
                    age_group = categorize_age_groups(age)
                    age_groups[age_group]['values'].append(val)
                    age_groups[age_group]['demographics'].append(demo)
                    if slice_interp is not None and idx < len(slice_interp):
                        age_groups[age_group]['slice_interp'].append(slice_interp[idx])

                grouped_by_sex[struc][struc_name][metric] = sex_groups
                grouped_by_age[struc][struc_name][metric] = age_groups
    
    return grouped_by_sex, grouped_by_age


def plot_metrics_by_sex(grouped_data, output_folder):
    """
    Create plots showing metrics by sex groups in a single subplot figure.
    
    Args:
        grouped_data: Dictionary grouped by sex from group_data_by_demographics
        output_folder: Path to save plots
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Define colors for sex groups
    color_map = {
        'Male': "#1f77b4",
        'Female': '#d62728'
    }
    
    groups = ['Male', 'Female']
    
    # Process each structure separately
    for struc in ['discs', 'foramens', 'vertebrae']:
        # Get all metrics and structures for this structure type
        all_structures = list(grouped_data[struc].keys())
        if not all_structures:
            continue
            
        # Get all unique metrics across all structures
        all_metrics = set()
        for struc_name in all_structures:
            all_metrics.update(grouped_data[struc][struc_name].keys())
        all_metrics = sorted(list(all_metrics))
        
        if not all_metrics:
            continue
        
        # Create subplot grid - one subplot per metric
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_metrics > 1 else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Structure {struc} - All Metrics by Sex', fontsize=16, y=0.98)
        
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            # Get all structures that have this metric
            structure_levels = []
            for struc_name in grouped_data[struc].keys():
                if metric in grouped_data[struc][struc_name]:
                    structure_levels.append(struc_name)
            
            # Filter structures based on anatomical range (start from T11/T10-T11)
            if struc == 'vertebrae':
                structure_levels = filter_structures_by_range(structure_levels, 'vertebrae', 'T11', 'L5')
            elif struc == 'discs':
                structure_levels = filter_structures_by_range(structure_levels, 'discs', 'T10-T11', 'L5-S1')
            elif struc == 'foramens':
                structure_levels = filter_structures_by_range(structure_levels, 'foramens', 'T10-T11', 'L5-S1')
            
            # Sort structures in anatomical order
            structure_levels = sort_anatomical_structures(structure_levels)
            
            if not structure_levels:
                continue
                            
            # Prepare data for line plot with error bars
            plot_data = {group: {'means': [], 'stds': [], 'levels': []} for group in groups}
            
            for level in structure_levels:
                for group in groups:
                    if level in grouped_data[struc] and metric in grouped_data[struc][level]:
                        values = grouped_data[struc][level][metric][group]['values']
                        if values:
                            # Handle nested lists (canal/CSF data) vs single values (other structures)
                            if isinstance(values[0], list):
                                # Flatten nested lists for canal/CSF data
                                flattened_values = []
                                for val_list in values:
                                    if isinstance(val_list, list):
                                        flattened_values.extend([v for v in val_list if not np.isnan(v) and v != -1])
                                    else:
                                        if not np.isnan(val_list) and val_list != -1:
                                            flattened_values.append(val_list)
                                
                                if flattened_values:
                                    plot_data[group]['means'].append(np.mean(flattened_values))
                                    plot_data[group]['stds'].append(np.std(flattened_values))
                                else:
                                    plot_data[group]['means'].append(np.nan)
                                    plot_data[group]['stds'].append(np.nan)
                            else:
                                # Handle single values for other structures
                                clean_values = [v for v in values if not np.isnan(v) and v != -1]
                                if clean_values:
                                    plot_data[group]['means'].append(np.mean(clean_values))
                                    plot_data[group]['stds'].append(np.std(clean_values))
                                else:
                                    plot_data[group]['means'].append(np.nan)
                                    plot_data[group]['stds'].append(np.nan)
                            plot_data[group]['levels'].append(level)
                        else:
                            plot_data[group]['means'].append(np.nan)
                            plot_data[group]['stds'].append(np.nan)
                            plot_data[group]['levels'].append(level)
            
            # Plot lines with error bars for each group
            x_positions = range(len(structure_levels))
            
            for group in groups:
                if plot_data[group]['means']:
                    means = plot_data[group]['means']
                    stds = plot_data[group]['stds']
                    
                    # Remove NaN values for plotting
                    valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                    if valid_indices:
                        valid_x = [x_positions[i] for i in valid_indices]
                        valid_means = [means[i] for i in valid_indices]
                        valid_stds = [stds[i] for i in valid_indices]
                        
                        # Plot line with error bars
                        ax.errorbar(valid_x, valid_means, yerr=valid_stds, 
                                   label=group, 
                                   color=color_map[group], 
                                   marker='o', linewidth=2, markersize=6,
                                   capsize=5, capthick=2, alpha=0.8)
            
            # Formatting
            ax.set_xticks(x_positions)
            ax.set_xticklabels(structure_levels, rotation=45, ha='right')
            ax.set_xlabel('Structure Level')
            ax.set_ylabel(f'{metric}')
            ax.set_title(f'{metric}')
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only add legend to first subplot
                ax.legend()
        
        # Remove empty subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_folder / f'{struc}_all_metrics_by_sex.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sex plot for structure: {struc}")


def plot_metrics_by_age(grouped_data, output_folder):
    """
    Create plots showing metrics by age groups in a single subplot figure.
    
    Args:
        grouped_data: Dictionary grouped by age from group_data_by_demographics
        output_folder: Path to save plots
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Define colors for age groups
    color_map = {
        '18-39': "#2ca02c",
        '40-59': '#ff7f0e',
        '60+': '#9467bd'
    }
    
    groups = ['18-39', '40-59', '60+']
    
    # Process each structure separately
    for struc in ['discs', 'foramens', 'vertebrae']:
        # Get all metrics and structures for this structure type
        all_structures = list(grouped_data[struc].keys())
        if not all_structures:
            continue
            
        # Get all unique metrics across all structures
        all_metrics = set()
        for struc_name in all_structures:
            all_metrics.update(grouped_data[struc][struc_name].keys())
        all_metrics = sorted(list(all_metrics))
        
        if not all_metrics:
            continue
        
        # Create subplot grid - one subplot per metric
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_metrics > 1 else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Structure {struc} - All Metrics by Age', fontsize=16, y=0.98)
        
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            # Get all structures that have this metric
            structure_levels = []
            for struc_name in grouped_data[struc].keys():
                if metric in grouped_data[struc][struc_name]:
                    structure_levels.append(struc_name)
            
            # Filter structures based on anatomical range (start from T11/T10-T11)
            if struc == 'vertebrae':
                structure_levels = filter_structures_by_range(structure_levels, 'vertebrae', 'T11', 'L5')
            elif struc == 'discs':
                structure_levels = filter_structures_by_range(structure_levels, 'discs', 'T10-T11', 'L5-S1')
            elif struc == 'foramens':
                structure_levels = filter_structures_by_range(structure_levels, 'foramens', 'T10-T11', 'L5-S1')
            
            # Sort structures in anatomical order
            structure_levels = sort_anatomical_structures(structure_levels)
            
            if not structure_levels:
                continue
                            
            # Prepare data for line plot with error bars
            plot_data = {group: {'means': [], 'stds': [], 'levels': []} for group in groups}
            
            for level in structure_levels:
                for group in groups:
                    if level in grouped_data[struc] and metric in grouped_data[struc][level]:
                        values = grouped_data[struc][level][metric][group]['values']
                        if values:
                            # Handle nested lists (canal/CSF data) vs single values (other structures)
                            if isinstance(values[0], list):
                                # Flatten nested lists for canal/CSF data
                                flattened_values = []
                                for val_list in values:
                                    if isinstance(val_list, list):
                                        flattened_values.extend([v for v in val_list if not np.isnan(v) and v != -1])
                                    else:
                                        if not np.isnan(val_list) and val_list != -1:
                                            flattened_values.append(val_list)
                                
                                if flattened_values:
                                    plot_data[group]['means'].append(np.mean(flattened_values))
                                    plot_data[group]['stds'].append(np.std(flattened_values))
                                else:
                                    plot_data[group]['means'].append(np.nan)
                                    plot_data[group]['stds'].append(np.nan)
                            else:
                                # Handle single values for other structures
                                clean_values = [v for v in values if not np.isnan(v) and v != -1]
                                if clean_values:
                                    plot_data[group]['means'].append(np.mean(clean_values))
                                    plot_data[group]['stds'].append(np.std(clean_values))
                                else:
                                    plot_data[group]['means'].append(np.nan)
                                    plot_data[group]['stds'].append(np.nan)
                            plot_data[group]['levels'].append(level)
                        else:
                            plot_data[group]['means'].append(np.nan)
                            plot_data[group]['stds'].append(np.nan)
                            plot_data[group]['levels'].append(level)
            
            # Plot lines with error bars for each group
            x_positions = range(len(structure_levels))
            
            for group in groups:
                if plot_data[group]['means']:
                    means = plot_data[group]['means']
                    stds = plot_data[group]['stds']
                    
                    # Remove NaN values for plotting
                    valid_indices = [i for i, m in enumerate(means) if not np.isnan(m)]
                    if valid_indices:
                        valid_x = [x_positions[i] for i in valid_indices]
                        valid_means = [means[i] for i in valid_indices]
                        valid_stds = [stds[i] for i in valid_indices]
                        
                        # Plot line with error bars
                        ax.errorbar(valid_x, valid_means, yerr=valid_stds, 
                                   label=f'{group} years', 
                                   color=color_map[group], 
                                   marker='s', linewidth=2, markersize=6,
                                   capsize=5, capthick=2, alpha=0.8)
            
            # Formatting
            ax.set_xticks(x_positions)
            ax.set_xticklabels(structure_levels, rotation=45, ha='right')
            ax.set_xlabel('Structure Level')
            ax.set_ylabel(f'{metric}')
            ax.set_title(f'{metric}')
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only add legend to first subplot
                ax.legend()
        
        # Remove empty subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_folder / f'{struc}_all_metrics_by_age.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved age plot for structure: {struc}")


def plot_canal_metrics_by_sex(all_values_df, discs_gap, last_disc, output_folder):
    """
    Create plots showing canal metrics by sex groups using seaborn lineplot.
    Shows data from T11 to L5 vertebrae range only.
    
    Args:
        grouped_data: Dictionary grouped by sex from group_data_by_demographics
        discs_gap: Gap between discs in slice indices
        last_disc: Name of the last disc
        output_folder: Path to save plots
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Define colors for sex groups
    color_map = {
        'Male': "#1f77b4",
        'Female': '#d62728'
    }
    
    metrics_dict = {
            'discs': ['median_thickness', 'DHI', 'volume', 'eccentricity', 'solidity'],
            'vertebrae': ['median_thickness', 'AP_thickness', 'volume'],
            'foramens': ['right_surface', 'left_surface', 'asymmetry_R-L'],
            'canal': ['area', 'diameter_AP', 'diameter_RL', 'eccentricity', 'solidity'],
        }
    
    # Process canal and CSF structures
    for struc in ['canal']:
        all_metrics = metrics_dict[struc]
        
        # Create subplot grid - one subplot per metric
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_metrics > 1 else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Structure {struc} - All Metrics by Sex', fontsize=16, y=0.98)
        
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            # Keep lines with metrics line equal to metric
            all_values_data = all_values_df[struc]['canal'][metric]

            # Use seaborn line plot
            sns.lineplot(
                x='slice_interp', 
                y='values', 
                hue='sex', 
                data=all_values_data, 
                palette=color_map,
                ax=ax, 
                errorbar='sd')
                
            # Generate disc positions for x-axis labels
            disc = last_disc
            nb_discs = all_values_data['slice_interp'].max()//discs_gap
            top_pos = discs_gap*(nb_discs+1)

            for i in range(nb_discs-1):
                top_vert = disc.split('-')[0]
                ax.axvline(x=top_pos, color='gray', linestyle='--', alpha=0.5)
                ax.text(top_pos - discs_gap//2, ax.get_ylim()[0], top_vert, verticalalignment='bottom', horizontalalignment='center', fontsize=12, color='black', alpha=0.7)
                top_pos -= discs_gap
                disc = previous_structure(disc)
            
            ax.set_xticks([])
            ax.set_xticklabels([])  # Hide x-tick labels to avoid clutter
            ax.set_xlim(60, all_values_data['slice_interp'].max())

            # Formatting
            ax.set_xlabel('Disc Level')
            ax.set_ylabel(f'{metric}')
            #ax.set_title(f'{metric}')
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only add legend to first subplot
                ax.legend(title='Sex')

        
        # Remove empty subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_folder / f'{struc}_all_metrics_by_sex.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sex canal plot for structure: {struc}")


def plot_canal_metrics_by_age(all_values_df, discs_gap, last_disc, output_folder):
    """
    Create plots showing canal metrics by age groups using seaborn lineplot.
    Shows data from T11 to L5 vertebrae range only.
    
    Args:
        grouped_data: Dictionary grouped by age from group_data_by_demographics
        discs_gap: Gap between discs in slice indices
        last_disc: Name of the last disc
        output_folder: Path to save plots
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Define colors for age groups
    color_map = {
        '18-39': "#2ca02c",
        '40-59': '#ff7f0e',
        '60+': '#9467bd'
    }
    metrics_dict = {
        'discs': ['median_thickness', 'DHI', 'volume', 'eccentricity', 'solidity'],
        'vertebrae': ['median_thickness', 'AP_thickness', 'volume'],
        'foramens': ['right_surface', 'left_surface', 'asymmetry_R-L'],
        'canal': ['area', 'diameter_AP', 'diameter_RL', 'eccentricity', 'solidity'],
    }
        
    # Process canal and CSF structures
    for struc in ['canal']:
        all_metrics = metrics_dict[struc]
        
        # Create subplot grid - one subplot per metric
        n_metrics = len(all_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_metrics > 1 else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Structure {struc} - All Metrics by Sex', fontsize=16, y=0.98)
        
        for idx, metric in enumerate(all_metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            # Keep lines with metrics line equal to metric
            all_values_data = all_values_df[struc]['canal'][metric]
            
            # Use seaborn line plot
            sns.lineplot(
                x='slice_interp', 
                y='values', 
                hue='age', 
                data=all_values_data, 
                palette=color_map,
                ax=ax, 
                errorbar='sd')
            
            # Generate disc positions for x-axis labels
            disc = last_disc
            nb_discs = all_values_data['slice_interp'].max()//discs_gap
            top_pos = discs_gap*(nb_discs+1)

            for i in range(nb_discs-1):
                top_vert = disc.split('-')[0]
                ax.axvline(x=top_pos, color='gray', linestyle='--', alpha=0.5)
                ax.text(top_pos - discs_gap//2, ax.get_ylim()[0], top_vert, verticalalignment='bottom', horizontalalignment='center', fontsize=12, color='black', alpha=0.7)
                top_pos -= discs_gap
                disc = previous_structure(disc)

            ax.set_xticks([])
            ax.set_xticklabels([])  # Hide x-tick labels to avoid clutter
            ax.set_xlim(60, all_values_data['slice_interp'].max())

            # Formatting
            ax.set_xlabel('Disc Level')
            ax.set_ylabel(f'{metric}')
            #ax.set_title(f'{metric}')
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only add legend to first subplot
                ax.legend(title='Age Group')
        
        # Remove empty subplots
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_folder / f'{struc}_all_metrics_by_age.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved age canal plot for structure: {struc}")

if __name__ == "__main__":
    main()