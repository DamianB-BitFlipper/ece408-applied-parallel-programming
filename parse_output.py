#!/opt/conda/bin/python

import sys
import re
import json

STDOUT_IDENTIFIER = "-" * 15 + " STDOUT " + "-" * 15

def print_help(script_name: str):
    print(f"Usage: {script_name} <logging_id>")

def main():
    if len(sys.argv) == 2:
        logging_id = int(sys.argv[1])
    else:
        print_help(sys.argv[0])
        sys.exit(1)


    # Read from stdin
    input_text = sys.stdin.read()

    # Final all of the test after the `STDOUT_IDENTIFIER`
    match = re.search(f"{STDOUT_IDENTIFIER}(.*)", input_text, re.DOTALL)

    # If there is no match, then print an error message
    if match is None:
        print("No match found")
        sys.exit(1)

    # Extract the text after the `STDOUT_IDENTIFIER`
    run_results_json = match.group(1)
    run_results = json.loads(run_results_json)

    for log_element in run_results["timer"]["elements"]:
        if log_element["id"] == logging_id:
            json.dump(log_element, sys.stdout, indent=2)
            return

    # If the logging ID was not found, then print an error message
    print("Logging ID not found")
    sys.exit(1)

if __name__ == "__main__":
    main()
