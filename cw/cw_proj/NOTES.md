# Notes on issues

## Issues
SGD Tuning

When using the Schosastic Gradient Decent there were issues encountered with the parameters, after following a tutorial on [tensorflow.org](https://www.tensorflow.org/tutorials/sequences/text_generation).

After looking at the documentation for SGD it was noticed that the normal parameter for moment was 0.0, whereas values tried were 0.1, 0.001 and so on.

## Adding more frames:
```
 cp -r $NTU_DIR/nturgb+d_depth/S001C001P004R001A* $NTU_DIR/{train/test}
```
 - Remember to update `folders_in_train/test.txt` file.
