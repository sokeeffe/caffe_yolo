#!/bin/bash
scriptName=$(basename "$0")

function stripLocal () {
    if [[ $1 == *.* ]]; then
        echo "$1" | cut -d. -f1
    else
        echo "$1"
    fi
}

if [ -z $2 ] ; then
    echo "*** ${scriptName} ERROR: called with args: $*"
    echo "usage ${scriptName} net_cfg net_weights"
    exit 1 
fi

netCfg=$1
netWeights=$2
finalOutFile=$(basename $2)
finalOutFile=`stripLocal $finalOutFile`
outFile=$(basename $3)
outFile=`stripLocal $outFile`

# python scripts/yolov2/valid_microyolo_v2.py \
#               ../datasets/SPLRegionOutput/val.txt ${netCfg} ${netWeights}

ballApp="ball.txt"
robotApp="robot.txt"
goalpostApp="goalpost.txt"
penspotApp="penspot.txt"
ballFile=$finalOutFile$ballApp
robotFile=$finalOutFile$robotApp
goalpostFile=$finalOutFile$goalpostApp
penspotFile=$finalOutFile$penspotApp

cp "ball_detects.txt" "results/$ballFile"
cp "robot_detects.txt" "results/$robotFile"
cp "goal_detects.txt" "results/$goalpostFile"
cp "pen_detects.txt" "results/$penspotFile"

python scripts/yolov2/meanAveragePrecision_spl_relative_region_debug.py "results/$ballFile" \
            spl_reg_gt_val.csv region_missed_detects.csv 0 "results/$finalOutFile""_ball_pr.csv"
python scripts/yolov2/meanAveragePrecision_spl_relative_region_debug.py "results/$robotFile" \
            spl_reg_gt_val.csv region_missed_detects.csv 1 "results/$finalOutFile""_robot_pr.csv"
python scripts/yolov2/meanAveragePrecision_spl_relative_region_debug.py "results/$goalpostFile" \
            spl_reg_gt_val.csv region_missed_detects.csv 2 "results/$finalOutFile""_goalpost_pr.csv"
python scripts/yolov2/meanAveragePrecision_spl_relative_region_debug.py "results/$penspotFile" \
            spl_reg_gt_val.csv region_missed_detects.csv 3 "results/$finalOutFile""_penspot_pr.csv"

python scripts/yolov2/map_yolo_debug.py "results/$finalOutFile""_ball_pr.csv" \
            "results/$finalOutFile""_robot_pr.csv" \
            "results/$finalOutFile""_goalpost_pr.csv" \
            "results/$finalOutFile""_penspot_pr.csv" \
            "results/$finalOutFile""_map.txt"
