#!/usr/bin/env bash

PROJECT_NAME="Core Tools"

echo ""
echo "Welcome to the $PROJECT_NAME installer"
echo ""

# Confirm user wants to install project
echo "This installer assumes the user has installed ROS Indigo and has properly
setup a Catkin workspace."
echo ""
echo "Please visit http://wiki.ros.org/indigo/Installation/
for more information on installing ROS Indigo and
http://wiki.ros.org/catkin/Tutorials/create_a_workspace on creating a Catkin
workspace."
echo ""

echo "Are you sure you want to install $PROJECT_NAME? [y/n]"
read user_input
while [ $user_input != "y" ]; do
  if [ $user_input == "n" ]; then
    exit 1
  fi
  echo "Please choose [y/n]"
  read user_input
done


# Define system packages required and Python packages required
SYSTEM_DEPENDENCIES=("software-properties-common"
                      "python-pip")
SYSTEM_DEPENDENCIES_NUM=${#SYSTEM_DEPENDENCIES[@]}

PYTHON_DEPENDENCIES=("PyYAML"
                      "scipy")
PYTHON_DEPENDENCIES_NUM=${#PYTHON_DEPENDENCIES[@]}

REPOSITORY_LINK=""

# Install required system packages using apt-get
count=1
echo ""
echo "Installing required system packages..."
for pkg in ${SYSTEM_DEPENDENCIES[@]}; do
  echo ""
  echo "Attempting to install $pkg ($count of $SYSTEM_DEPENDENCIES_NUM)"
  sudo apt-get install $pkg
  count=$((count+1))
done

# Install required Python packages using pip
count=1
echo ""
echo "Installing required Python packages..."
for pkg in ${PYTHON_DEPENDENCIES[@]}; do
  echo ""
  echo "Attempting to install $pkg ($count of $PYTHON_DEPENDENCIES_NUM)"
  pip install $pkg
  count=$((count+1))
done

echo "Install finished. Exiting..."
exit 0
