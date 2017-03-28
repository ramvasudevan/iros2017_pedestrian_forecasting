## Synopsis

This project seeks to predict pedestrian position seconds in the future with a millisecond time-budget.  The motivation for this project is for inclusion in autonomous vehicle software, which requires real-time forecasting of pedestrian actions.

## Files


## Usage

### How to use this library###


#### Introduction ####

This document is intended to illustrate how to use this code for the prediction as illustrated in our [paper](google.com).

It should be fairly self evident from this document how this is implemented. If you find errors, unclear sections in this document, or OS specific problems with installation, please contact me at [owhughes@umich.edu](mailto:owhughes@umich.edu)

#### Installation gotchas ####
Ensure that `scipy` is the most recent version available on your OS.


#### How to use ####


##### Making sure your data is formatted correctly #####

We used the Stanford Drone Dataset as our test data, so we use their format which can be observed in the ```annotations``` folder. It describes observations in terms of a bounding box around the pedestrian. The column labels are:

```id, x1, y1,x2, y2, frame, lost, occluded, generated, label```

We don't pay attention to `lost, occluded,` or `generated`. 

`id` is a unique identifier corresponding to a pedestrian. `x1, y1` is the lower-left corner of the bounding box. `x2, y2` is the upper right. `frame` corresponds to the frame number of the observation, and `label` corresponds to the type of agent observed. Our model defaultly reads `Pedestrian` labels, but you can set this yourself (see below). The spatial coordinates are measured in pixels, so express it in integers.

Put your data in a file called `annotations.txt`.

##### Learn a scene #####

In the same folder as `annotations`, make a file called `params.json`. The file should look something like 

```
{
   "width": 100,
   "height": 100,
   "label": "Pedestrian",
   "prefix": "test"
}
```

`width` is the width of the scene in pixels, `height` is the height in pixels, `label` is as above, `prefix` is what the pickle files should be prefixed with when the learned scene is saved.

Then run `python make_scene.py path/to/folder`, and pass the path to the folder containing `annotations.txt` and `params.json`. If your data is well formatted, it should build a scene and put it in the `scene_folder` specified in `config.json`.



## Contributors

Henry O. Jacobs

## License
