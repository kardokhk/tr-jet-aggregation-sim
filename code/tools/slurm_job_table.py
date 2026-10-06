"""Print a markdown table of the Slurm jobs of this project from the cluster accounting records.

Usage (login node; Python 3.6 or later, standard library only):
    python3 code/tools/slurm_job_table.py [--since 2026-09-18] [--pattern 'duomax|osrev']

One row per job submitted from the current working directory whose name matches the pattern. The cluster
bills the whole node: node-hours = elapsed time x 1 node. sacct counts 256 hardware threads per node, so
its CPU-hours are twice the core-hours on the 128 physical cores.
"""
from __future__ import print_function

import argparse
import getpass
import os
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default="2026-09-18")
    ap.add_argument("--pattern", default="duomax|osrev")
    a = ap.parse_args()
    out = subprocess.check_output(["sacct", "-u", getpass.getuser(), "-S", a.since, "-X", "-n", "-P",
                                   "--format=JobID,JobName,State,Elapsed,CPUTimeRAW,NodeList,Start,WorkDir"])
    print("| Job | Name | Start | Node | State | Elapsed | Node-hours | CPU-hours (sacct) | Core-hours |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = [0.0, 0.0, 0.0]
    for line in out.decode().strip().split("\n"):
        job, name, state, elapsed, cpu, node, start, wd = line.split("|")
        if os.path.realpath(wd) != os.path.realpath(os.getcwd()) or not re.search(a.pattern, name):
            continue
        h, m, s = [int(x) for x in elapsed.split(":")]
        nh, ch = (h * 3600 + m * 60 + s) / 3600.0, int(cpu) / 3600.0
        tot = [tot[0] + nh, tot[1] + ch, tot[2] + ch / 2]
        print("| %s | %s | %s | %s | %s | %s | %.3f | %.2f | %.2f |"
              % (job, name, start.replace("T", " "), node, state.split()[0], elapsed, nh, ch, ch / 2))
    print("| total | | | | | | %.3f | %.2f | %.2f |" % tuple(tot))


if __name__ == "__main__":
    main()
