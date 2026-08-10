# Trajectory Label Definitions

This document defines the four pitch-trajectory shape classes used in the dataset. Each section describes the expected shape, positive and ambiguous cases, and exclusion criteria. Refine these definitions as annotation proceeds.

---

## trajectory_1 (Fixed)

### Description

A sustained pitch with minimal intentional movement. The performer holds a stable swara without a pronounced bend or glide. In idtap terminology this corresponds to a **Fixed** trajectory.

### Start behavior

The pitch reaches the target swara quickly and settles with little or no initial glide.

### Middle behavior

The pitch remains largely constant through the middle of the segment, with only minor natural vibrato or intonation variation.

### End behavior

The pitch ends at approximately the same level as the sustained middle portion, without a directional bend into the next note.

### Positive examples

- Steady held notes in alap or slow gat passages
- Clearly fixed swaras where pitch contour is flat relative to local tonic

### Ambiguous examples

- Fixed notes with heavy natural vibrato that could resemble a shallow bend
- Very short notes where start/end behavior is hard to judge

### Exclusion criteria

- Segments containing an intentional glide or bend at start or end
- Passages with overlapping notes from accompaniment dominating the pitch track
- Segments where the annotator cannot confidently identify a single sustained swara

---

## trajectory_2 (Bend: Simple)

### Description

A single continuous bend between two pitch levels without a distinct sloped plateau at the start or end. In idtap terminology this corresponds to a **Bend: Simple** trajectory.

### Start behavior

The pitch begins at one stable level and moves smoothly toward the target level.

### Middle behavior

The transition is relatively uniform; there is no extended flat region before or after the main bend.

### End behavior

The pitch arrives at the target level and stabilizes, or the bend continues through the segment boundary without a separate sloped-end phase.

### Positive examples

- Classic meend between adjacent swaras with a smooth arc
- Single-direction glides that do not pause on a sloped ramp at either end

### Ambiguous examples

- Bends that are very slow and could be labeled as sloped start or sloped end
- Short bends where only part of the trajectory is visible in the clip

### Exclusion criteria

- Trajectories with a clear flat onset followed by bend (see trajectory_3)
- Trajectories with a clear flat ending after a bend (see trajectory_4)
- Multi-segment ornaments that combine several distinct shapes

---

## trajectory_3 (Bend: Sloped Start)

### Description

A bend that begins with a sloped or ramped approach before the main transition. In idtap terminology this corresponds to a **Bend: Sloped Start** trajectory.

### Start behavior

The pitch moves gradually from the initial level along a sloped ramp rather than departing immediately from a fixed starting point.

### Middle behavior

After the sloped onset, the bend may continue smoothly toward the target pitch.

### End behavior

The pitch typically stabilizes at or near the target level, without a distinct sloped-end plateau (otherwise consider trajectory_4).

### Positive examples

- Meends where the performer “eases into” the bend from a rising or falling ramp
- Ornaments where the opening motion is clearly sloped before the main bend

### Ambiguous examples

- Gentle simple bends that could be interpreted as a shallow sloped start
- Noisy pitch tracks where the slope at the onset is unclear

### Exclusion criteria

- Fixed notes with no meaningful onset slope
- Bends dominated by a sloped ending rather than a sloped start
- Segments where start behavior is truncated by clip boundaries

---

## trajectory_4 (Bend: Sloped End)

### Description

A bend that finishes with a sloped or ramped approach to the final pitch level. In idtap terminology this corresponds to a **Bend: Sloped End** trajectory.

### Start behavior

The pitch may begin at a fixed level or already be in motion from an earlier bend.

### Middle behavior

The main bend occurs before the final sloped approach to the target.

### End behavior

The pitch approaches the final level along a sloped ramp rather than snapping to a fixed ending.

### Positive examples

- Meends that taper into the destination swara with a visible end ramp
- Ornaments where release into the final note is sloped rather than abrupt

### Ambiguous examples

- Simple bends with a long decay that could look sloped at the end
- Clips cut before the full sloped ending is complete

### Exclusion criteria

- Fixed endings with no sloped release
- Bends where the dominant characteristic is a sloped start (see trajectory_3)
- Segments where end behavior is truncated by clip boundaries
