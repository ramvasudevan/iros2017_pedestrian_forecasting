How to use this library
========

Introduction
---------
This document is intended to illustrate how to use this code for the prediction as illustrated in our [paper](google.com).

It should be fairly self evident from this document how this is implemented. If you find errors, unclear sections in this document, or OS specific problems with installation, please contact me at [owhughes@umich.edu](mailto:owhughes@umich.edu)

Installation gotchas
------
Ensure that ```scipy``` is the most recent version available on your OS.


How to use
-----

###Making sure your data is formatted correctly ###

We used the Stanford Drone Dataset as our test data, so we use their format which can be observed in the ```annotations``` folder.The column labels are:

<center> ```id, x1, y1,x2, y2, frame, lost, occluded, generated, label``` </center>



