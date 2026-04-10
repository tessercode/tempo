# Audio Preprocessing
These are my notes on how to prepare CW recordings for use in the operator parameter validation experiment (exp4). 

## Obtaining Recordings
The CW recordings used in this work are publicly available. Written permission was obtained from the dataset maintainers to use them in this publication, but the recordings themselves are available for anyone to download.

### Italian CW Recordings
Written permission to use these recordings in this publication was granted by archive holder Luigi Ciampoli (IZ4KBW), and featured operator Luigi Mastroianni (IZ8VKW)

[https://swling.com/blog/2021/11/large-archive-of-cw-off-air-recordings-from-italy/](https://swling.com/blog/2021/11/large-archive-of-cw-off-air-recordings-from-italy/)

### QRQcw Net Recordings
Written permission to use these recordings in this publication was granted by Charles Vaughn (AA0HW), maintainer of the QRQcw archive.

[https://qrqcwnet.ning.com/page/theodore-roosevelt-mcelroy](https://qrqcwnet.ning.com/page/theodore-roosevelt-mcelroy)

## Labelling Transmissions for Analysis

For multi-operator net recordings, automatic message boundary detection is unreliable when operators differ significantly in signal level. Use Audacity's Label Track to manually mark each operator's transmission before running the analysis pipeline.

1. Open the recording in Audacity 

2. Add a Label Track: Tracks → Add New → Label Track.

3. Select each operator's transmission (click and drag), press Ctrl+B to create a label, and type the operator's callsign or a short identifier (e.g. `IT9ETC`, `msg1`).

4. Select each labelled message segment and normalize to -1db with DC bias removal

5. Export the normalized recordings: File → Export → Export wav. Save the file alongside the audio file with the same base name (e.g. `recording_name.wav` next to `recording_name.mp3`). 

6. Export the labels: File → Export → Export Labels. Save the file alongside the audio file with the same base name (e.g. `recording_name.txt` next to `recording_name.mp3`). The exported format is tab-separated: `start_time`, `end_time`, `label`.

7. Add the file paths to the recording's entry in `RECORDING_REGISTRY` in the notebook:
   ```python
   'path': 'data/recording_name.wav',
   'label_path': 'data/recording_name.txt',
   ```
   The batch runner will automatically use `analyze_labeled_recording()` when a label file is present, and fall back to automatic boundary detection otherwise.

### Why this is necessary

The adaptive threshold in the analysis pipeline is calibrated per segment after RMS normalisation. If automatic boundary detection mistakes a quiet operator's transmission for silence, that transmission is never extracted as a segment and normalisation never runs on it. Manually marking boundaries in Audacity — where the waveform and spectrogram are directly visible — is more reliable than any energy-based silence detector for typical HF net recordings. This allows for significantly better operator statistics as well because it allows for statistics to be reliably collected at the message / operator level instead of at the recording level.