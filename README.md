# Integrating DFL into VFX production

## New features in main scripts:
**Extractor**
- bit-depth agnostic: high bit-depth images are read without issue; saved alignements are still 8bit (but aligned images mostly only exist for storing landmark data)
- implementation of [landmark stabilization](https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/) is on by default (n=5), number of detection iterations can be modified by passing `--stabilize-n` argument

**Merger**
- bit-depth agnostic: high bit-depth images are read without issue, merge output is written same-as-source
- "batch mode" for merging on render farm is enabled with the following new arguments:
  - `--frame-range` Frame range to process. Can be single frame, dash-separated range, or comma-separated list of either 
  - `--cfg-load-path` Load a saved configuration for merging with no prompts
  - `--cfg-save-path` Save a configuration file only for later use

## New scripts:
**`filter.py`** Temporal filtering of landmarks. Missing frames are interpoloated from surrounding frames and existing frames are filtered in time to smooth alignments and reduce jitter which often negatively affects merges. Use with the typical required arguments:
  - `--input-dir` Path to the full shot plate
  - `--aligned-dir` Path aligned files created by earlier extraction
  - `--output-dir` Path to the save new filtered alignments
  <br>Optional Arguments:
  - `--radius` Number of frames before and after current over which to integrate
  - `--sigma'` Std deviation for filter kernel - higher value has stronger effect
  - `--output-debug'` Output aligned debug images

**`warp.py`** Apply a warp to input frames based on landmarks (from alignment) and a deltas file containing vectors to warp landmarks. A deltas file will be unique to each src-dst pair. Creation of deltas files is currently happening in Nuke, but a simple UI is somewhere on the TODO list. Otherwise, it takes the typical arguments:
  - `--input-dir` Input directory. A directory containing the files you wish to warp
  - `--aligned-dir` Path aligned files created by earlier extraction
  - `--output-dir` Path to the save the warped images
  - `--deltas-file` Path to file containing landmark deltas
  <br>Optional Arguments:
  - `--border` Amount of non-warped border around aligned face (in percent) - default 0"
  - `--blur` Size of blur to apply to warp map (in percent of image size) - default 5%"
  - `--raw` Operate on aligned image / raw predictions
  - `--save-map` Save warp map only, for later compositing step

**`extract_masks.py`** A quick and dirty tool to write rgb masks from landmarks. The masks are very low quality currently
  - `--aligned-dir` Path aligned files created by earlier extraction
  - `--output-dir` Path to the save the mask images
  - `--raw` Output masks in aligned space, default is screen space")
