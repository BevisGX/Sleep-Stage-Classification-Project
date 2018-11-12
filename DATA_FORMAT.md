#data format instruction

##sleep stage data
Sleep stage data should be stored in csv format which means elements should be separated by ','.

Since there are 5 sleep stage:
- wake
- N1
- N2
- N3
- unrecorded
- REM

So we define one line as one sample, that means one frame's stage, encoded by one-hot-encode.

For example, if one frame's stage is N2 and the the next frame's stage is REM, it should be recorded like:

0, 0, 1, 0, 0

0, 0, 0, 0, 1