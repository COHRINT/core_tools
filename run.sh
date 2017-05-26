#!/usr/bin/env bash

clear
tester="./main_tester.py"
wall_path="~/catkin_ws/src/cops_and_robots/src/cops_and_robots/map_tools/walls.json"
model_path="~/catkin_ws/src/cops_and_robots/src/cops_and_robots/map_tools/models.json"

#<>TODO: SANITIZE INPUT
#<>TODO: GENERALIZE PATHS TO MODEL FILES AND TESTER FILE
#<>TODO: IMPROVE INTERFACE AND MENU, HAVE MENU SHOW UP CONSISTENTLY

# Startup shell script for Core Tools testbed, including ROS stack
echo '--------------------------------------------------------------------'
echo '--------------------------------------------------------------------'
echo '--------------------- Core Tools Testbed ---------------------------'
echo '--------------------------------------------------------------------'
echo '--------------------------------------------------------------------'
echo ' '
echo '*** Experiment parameters can be changed in config.yaml ***'
echo '*** Maps are generated using new_map.py ***'
echo ' '

# echo "Testbed is set to run $tester. Enter [n] to specify path to different test file, or any other key to continue."
# read tester_input
# if [ $tester_input == "n" ]; then
#   echo "Specify new path:"
#   read tester_input
#   tester=$tester_input
# fi
# echo " "

echo 'Create new map? [y/n]'

read new_map
resolution=100

echo $1 $2 $3 $4

robots=("deckard" "roy" "pris" "zhora")
#1 indicates use, positons in arrays match, e.g. use[0]==1 means using deckard
# needs to match config.yaml!
#use=($1 $2 $3 $4)
use=(1 1 0 0)
count=0
if [ $new_map == "y" ]; then
  echo 'Enter desired resolution'
  read resolution
  echo "Enter [y] to use current wall file: $wall_path OR enter new path to wall json file"
  read wall_path
  echo "Enter [y] to use current model file: $model_path OR enter new path to model json file"
  read model_path
  xterm -e "cd ~/catkin_ws/src/cops_and_robots/src/cops_and_robots/map_tools && \
              python new_map.py $resolution" &
  for i in ${robots[@]}; do
    #echo ${use[$count]}
    if [ ${use[$count]} -eq 1 ]; then
      xterm -e "scp ~/Desktop/occupancy_grid.png odroid@$i:~/cops_and_robots/resources/maps/OccupancyGrid" &
    fi
    count=$count+1
  done
fi

# start roscore and vicon_sys
xterm -e bash -c "roscore" &
sleep 2
xterm -e bash -c "roslaunch ~/vicon_sys.launch" &

echo "--------------------"
echo " "
echo "Log into all robots and run the command:"
echo "roslaunch cops_and_robots/launch/vicon_nav.launch"
echo " "
#echo "If a new map was created, transfer it to the robots using (in a new terminal window):"
#echo "scp ~/Desktop/occupancy_grid.png odroid@\$robot:/cops_and_robots/resources/maps/OccupancyGrid"

count=0
for i in ${robots[@]}; do
  connection_input=1
  if [ ${use[$count]} -eq 1 ]; then
    while [ $connection_input -ne 0 ]; do
        if [ $connection_input -eq 1 ] ; then
            xterm -e "ssh odroid@$i" &
        fi
        echo " "
        echo "--------------------"
        echo "Enter '0' if connection was successful"
        echo "Enter '1' to retry $i connection"
        read connection_input
    done
  fi
  count=$count+1
done

#run code for experiment
echo "When vicon_nav.launch has been started on all robots, press ENTER to run experiment"
read x
run_input=1
while [ $run_input -ne 0 ]
do
    if [ $run_input -eq 1 ]; then
      xterm -hold -e "python $tester"
      echo "-------"
      echo "Enter '1' to re-run experiment"
      echo "or enter '0' to end the program"
    fi
    read run_input
done
echo "wheeeeeeeeee about to exit"
exit 0
