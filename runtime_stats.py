#!/opt/conda/bin/python

import sys
import subprocess
import json

def print_help(script_name: str):
    print(f"Usage: {script_name} <n_iters> <subprocess_str>")


def main():
    if len(sys.argv) == 3:
        n_iters = int(sys.argv[1])
        subprocess_str = sys.argv[2]
    else:
        print_help(sys.argv[0])
        sys.exit(1)

    run_times = []
    for i in range(n_iters):
        result_json = subprocess.run(
            subprocess_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )

        # Print the stderr
        if result_json.stderr:
            print("-" * 15 + " STDERR " + "-" * 15)
            print(result_json.stderr)
            sys.exit(1)

        # Parse the JSON output
        result = json.loads(result_json.stdout)

        # The elapsed time is in nanoseconds, so divide by 1_000_000 to get ms
        print(f"{i:3d}: {result['elapsed_time'] / 1_000_000:.3f} ms")

        # Add the elapsed time to the list of run times
        run_times.append(result["elapsed_time"])

    # Print the average run time
    print("-" * 15 + " AVERAGE " + "-" * 15)
    print(f"Average: {(sum(run_times) / len(run_times)) / 1_000_000:.3f} ms")

    # Print the minimum run time
    print("-" * 15 + " MINIMUM " + "-" * 15)
    print(f"Minimum: {min(run_times) / 1_000_000:.3f} ms")

if __name__ == "__main__":
    main()
