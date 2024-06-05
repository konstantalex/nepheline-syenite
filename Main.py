import matplotlib.pyplot as plt

from src.analysis import StitchedSemProcessor, FractionProcessor
from src.config import parameters


#########################################################
# Main
#########################################################


def main():

    if parameters["run_fraction"]:
        stitched_proc = FractionProcessor()
        stitched_proc.run()

    if parameters["run_sem"]:
        sem_proc = StitchedSemProcessor()
        sem_proc.run()

    if parameters["display_plots"]:
        plt.show()

    print("\n\n\nPROGRAM FINISHED\n\n\n")


if __name__ == "__main__":
    main()
