"""
Top-level orchestration script. Uncomment the call(s) you want to run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.extractor import frame as extractor
from data_pipeline.extractor import retrieve_annotated_images as retrieve
from data_pipeline.extractor import retrieve_annotated_xml
from data_pipeline.extractor import sample as sample
from data_pipeline.annotator import convert_voc_to_yolo as convert
from data_pipeline.annotator import split_dataset_yolo as split
from data_pipeline.annotator import helper as annotator
from data_pipeline.analyzer import run_full_analysis as analysis, fix_csv
from yolo_baselines.run import main as yolo_main
from scripts.run_training import main as mob3_main
from scripts.run_baselines import run as run_baselines
from scripts.run_distillation import run_distillation

if __name__ == '__main__':
    ## this function extracts the frames from the folder data/raw/video
    # extractor()

    ## function to retrieve the images of the current labels annotated
    # retrieve(90)
    # retrieve_annotated_xml()

    ## this function samples frames for labeling
    # sample()

    ## function to train, and annotate
    # annotator()

    ## to make life easier, convert to YOLO format the labels
    # convert()
    # split()

    # metrics for dataset analysis
    # analysis()
    # fix_csv()



    ############################
    #                          #
    #      Neural networks     #
    #                          #
    ############################
    # Yolo baselines
    # yolo_main()

    # Custom model based on Mobilenetv1
    # mob1_main()
    #mob3_main()

    # training of neural network
    # train()

    #baselines (ablation)
    #run_baselines()

    #Distillation
    run_distillation()
