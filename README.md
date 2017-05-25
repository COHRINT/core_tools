# Core Tools for COHRINT Research

This repository hosts common tools and an experiment testbed used by multiple
experiments by the COHRINT lab group at CU Boulder's Aerospace Engineering
Sciences.

**Any contributions to this repository must be made with the understanding that
many projects pull from the same code base and you may break someone else's
experiment by contributing.**

## Testbed Content

The testbed contains tools for running experiments with the iRobot Create
hardware platform used by the COHRINT lab. The structure of an experiment using
the testbed consists of a main update loop in which goal planners update goal
points for the robots.

Included in the tools in the testbed are a user interface for configuring YAML
files to describe the parameters of an experiment, a user interface for making
observations, and a ROS service for manging the translation of policies into
goal points and updating beliefs.

Please see the Core Tools GitHub repository wiki located at
https://github.com/COHRINT/core_tools/wiki for more information and instructions
on using the testbed.
