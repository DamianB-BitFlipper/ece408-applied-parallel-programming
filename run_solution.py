#!/opt/conda/bin/python

import re
import sys
from pathlib import Path
import subprocess

def print_help(script_name: str):
    print(f'Usage: {script_name} <executable> <test_data_dir> <compute_type>')
    print('  executable: path to the executable to test')
    print('  test_data_dir: path to the directory containing the test data')
    print('  compute_type: type of the compute (vector, matrix or image)')

def get_input_file_id(filepath: Path) -> int:
    file_id = re.search(r'input(\d*)\.raw', filepath.name).group(1)
    if file_id == '':
        return 0
    else:
        return int(file_id)

def main():
    # Expect 3 arguments: executable to test, test data directory and compute type
    if len(sys.argv) == 4:
        executable = Path(sys.argv[1])
        test_data_dir = Path(sys.argv[2])
        compute_type = sys.argv[3]
    else:
        print_help(sys.argv[0])
        print(f'Error: expected 3 arguments, got {len(sys.argv) - 1}')
        sys.exit(1)

    # The `executable` and `test_data_dir` must exist
    if not executable.exists() or not test_data_dir.exists():
        print_help(sys.argv[0])
        print('Error: the executable or the test data directory does not exist')
        sys.exit(1)

    # The `executable` must be a file
    if not executable.is_file():
        print_help(sys.argv[0])
        print('Error: the executable must be a file')
        sys.exit(1)

    # The `test_data_dir` must be a directory
    if not test_data_dir.is_dir():
        print_help(sys.argv[0])
        print('Error: the test data directory must be a directory')
        sys.exit(1)

    # The `compute_type` must be one of the following: vector, matrix or image
    if compute_type not in {'vector', 'matrix', 'image'}:
        print_help(sys.argv[0])
        print('Error: the compute type must be one of the following: vector, matrix or image')
        sys.exit(1)

    # Combine the test data input filepaths
    input_filepaths = sorted(test_data_dir.glob(f'input*.raw'),
                             key=get_input_file_id)

     # Comma separate the `input_filepaths` into one string
    input_filepaths_str = ','.join(map(str, input_filepaths))

    # Get the output filepath
    output_filepath_str = str(test_data_dir / 'result.raw')

    # Get the expected output filepath
    expected_output_filepath_str = str(test_data_dir / 'output.raw')

    # Helpful information
    print(f"Running '{executable}' with the following arguments:")
    print(f"  input_filepaths: '{input_filepaths_str}'")
    print(f"  output_filepath: '{output_filepath_str}'")
    print(f"  expected_output_filepath: '{expected_output_filepath_str}'")
    print(f"  compute_type: '{compute_type}'")
    print()

    # Run the executable
    result = subprocess.run(
        [executable,
         "-i",
         input_filepaths_str,
         "-o",
         output_filepath_str,
         "-e",
         expected_output_filepath_str,
         "-t",
         compute_type
         ],
        check=True,
        text=True,
        capture_output=True,
    )

    # Print the stderr
    if result.stderr:
        print("-" * 15 + " STDERR " + "-" * 15)
        print(result.stderr)

    # Print the stdout
    print("-" * 15 + " STDOUT " + "-" * 15)
    print(result.stdout)

if __name__ == '__main__':
    main()
