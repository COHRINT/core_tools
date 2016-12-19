# ssh into robot: name passed in as command line argument
ROBOT=$1

echo "ssh into robots and run the command: roslaunch cops_and_robots/launch/vicon_nav.launch"
connection_input=3
while [ $connection_input -ne 0 ]
do
    if [ $connection_input -eq 1 ] ; then
        xterm -e "ssh odroid@$COP" &
    fi
    echo "-------"
    echo "Enter '0' if connection was successful"
    echo "Enter '1' to retry $ROBOT connection"
    read connection_input
done

exit 0
