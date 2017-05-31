#!/usr/bin/env bash

PROJECT_NAME="Cops and Robots 2.0"
FILE_NAME="cops_and_robots_2"

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
while [ "$user_input" != "y" ]; do
  if [ "$user_input" == "n" ]; then
    echo "Exiting installer..."
    exit 1
  fi
  echo "Please choose [y/n]"
  read user_input
done


# Define system packages required and Python packages required
SYSTEM_DEPENDENCIES=("python-pip"
                      "python-pyqt5")
SYSTEM_DEPENDENCIES_NUM=${#SYSTEM_DEPENDENCIES[@]}

PYTHON_DEPENDENCIES=("--upgrade pip"
                      "PyYAML"
                      "scipy"
                      "matplotlib"
                      "shapely"
                      "pandas"
                      "pytest"
                      "descartes"
                      "pyserial")
PYTHON_DEPENDENCIES_NUM=${#PYTHON_DEPENDENCIES[@]} # subtract 1 b/c --upgrade pip is counted as 2

REPOSITORY_LINK=""

# Install required system packages using apt-get
count=1
echo ""
echo "Installing required system packages..."
sudo apt-get update
sudo apt-get install software-properties-common
sudo apt-add-repository universe
sudo apt-get update
for pkg in ${SYSTEM_DEPENDENCIES[@]}; do
  echo ""
  echo "Attempting to install $pkg ($count of $SYSTEM_DEPENDENCIES_NUM)"
  sudo apt-get install $pkg
  count=$((count+1))
done

# install virtualenv  and create environment before installing python packages
echo "Installing virtualenv..."
sudo pip install virtualenv
echo ""
echo "Do you want to create a virtual environement for this project? (This is highly suggested)[y/n]"
read create_env

while [ "$create_env" != "y" ] && [ "$create_env" != "n" ]; do
  echo "Please choose [y/n]"
  read create_env
done

# create directory for virtual environment and then create environment
if [ "$create_env" == "y" ]; then
  env_name=$FILE_NAME
  dir_name="$HOME/.virtualenvs"
  if [ ! -d $dir_name ]; then
    echo "Creating directory $dir_name"
    mkdir $dir_name
  fi
  cd $dir_name
  virtualenv --system-site-packages $env_name
  echo "Sourcing environment..."
  source $dir_name/$env_name/bin/activate
  echo "Type the commmand 'source $dir_name/$env_name/bin/activate' to activate this environment."
  echo "Type 'deactivate' to deactivate the environment"
fi

# Install required Python packages using pip
count=1
echo ""
echo "Installing required Python packages..."
for pkg in "${PYTHON_DEPENDENCIES[@]}"; do
  echo ""
  echo "Attempting to install $pkg ($count of $PYTHON_DEPENDENCIES_NUM)"
  pip install --ignore-installed $pkg
  count=$((count+1))
done

echo "Install finished. Exiting..."
exit 0
