# data format instruction

## sleep stage data
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

## sleep event data
Sleep event data should be stored in csv format too.

The key include:
- events
- type
- start time(s)
- frame id
- duration(s)

And also one line as one sample, that means one event.

For example, one record could like this:

SpO2, RelativeDesaturation, 1423, 48, 65
