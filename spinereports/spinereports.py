import argparse
import multiprocessing as mp
from pathlib import Path
import textwrap

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