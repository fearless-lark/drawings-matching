# Drawings matching
Drawings matching has been developed leveragning two naive approaches:
- template matching with CCOEFF
- keypoint detection and matching with SIFT

## How to run
1. Place all the objects you want to match to the corresponding folder in `data/object`
2. Specify the path to the target image in the `config.yml'
3. Prepare and activate dedicated virtual environment via `venv` or `conda`
4. Install requirements via `python3 install -r requirements.txt`
5. Run `matching_ccoeff.py` to obtain matching results leveraging CCOEFF
6. Run `matching_sift.py` to obtain matching results leveraging SIFT
7. Check `output/` folder for the csv file with detections and png image with drawings
8. Play with parameters in the `config.yml` and choose the best parameters for your task

## Implementation details and some notes
The implementation is fairly straightforward.

After some trial and error, it was possible to get some meaningful matches via SIFT.

The one thing that boosted it the most was the angle_tolerance parameter. I introduced it in order to remove shapes that do not look like a rectangle (since our objects are very close to a rectangular shape). This helped to remove all false positives.
The rest was pretty trivial for matching tasks.

The rest of the stuff was pretty trivial for matching tasks and probably doesn't need to be mentioned.