import os
import sys
import time
import logging
from typing import Optional, Any, List, Dict, Set

import click
import pandas as pd
from tqdm import tqdm
import concurrent.futures

from snipe.api.enums import SigType
from snipe.api.sketch import SnipeSketch
from snipe.api.snipe_sig import SnipeSig
from snipe.api.reference_QC import ReferenceQC

from snipe.cli.cli_qc import qc as cli_qc
from snipe.cli.cli_sketch import sketch as cli_sketch
from snipe.cli.cli_ops import ops as cli_ops
from snipe.cli.cli_qc_parallel import parallel_qc

# pylint: disable=logging-fstring-interpolation

@click.group()
def cli():
    """
    Snipe CLI Tool

    Commands:
    - `sketch`: Perform sketching operations on genomic data.
    - `qc`: Execute quality control (QC) on multiple samples against a reference genome.
    - `ops`: Perform various operations on sketches.
    """
    pass


cli.add_command(cli_qc)
cli.add_command(cli_sketch)
cli.add_command(cli_ops)

if __name__ == '__main__':
    cli()
