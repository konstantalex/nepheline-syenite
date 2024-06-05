import os
from pathlib import Path

ROOT = Path(os.getcwd())
FOLDER_PATH = ROOT / "images"
FOLDER_FRACTION = FOLDER_PATH / "source_fraction"
FOLDER_SEM = FOLDER_PATH / "source_sem"
DESTINATION_FOLDER_SEM_PATCHED = FOLDER_PATH / "results_patched_sem"
DESTINATION_FOLDER_FRACTIONS = FOLDER_PATH / "results_fractions"
REPORT_FILE_PATH = DESTINATION_FOLDER_SEM_PATCHED / "report.txt"


##############################################
# PARAMETERS
##############################################

parameters = {
    "run_fraction": False,
    "run_sem": True,
    "display_plots": False,
    "show_images": False,
}

##############################################
