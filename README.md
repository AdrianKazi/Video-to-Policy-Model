# Action Inference

This project explores action inference from video without starting from human-defined action labels such as "walking", "standing", or "flying".

The core idea is to define an action mathematically as the dynamics of objects over time.

## Hypothesis

An action is not a word label. An action is a structured transformation of objects across frames.

Given a video sequence:

```text
I_t, I_{t+1}, ..., I_{t+k}
```

we first identify objects, represent each object as a set of geometric points, and define the action as the temporal evolution of those points.

## Object Representation

For each object in a frame, we want a geometry-aware representation:

```text
O_t = {p_1^t, p_2^t, ..., p_n^t}
```

where points may include:

- object centroid
- bounding box corners
- contour points
- high-curvature points
- bend or joint-like points for deformable subparts
- optional skeleton or medial-axis points

The number and placement of points should depend on the object shape, not on a fixed human semantic label.

## Action Representation

An action is represented as the motion and deformation of object points over time:

```text
A_t = O_{t+1} - O_t
```

or, for a longer temporal window:

```text
A_{t:t+k} = trajectory({p_i^t})
```

Useful features may include:

- displacement
- velocity
- acceleration
- rotation
- scale change
- deformation
- contact changes between objects
- relative motion between objects

This makes the action a latent geometric-dynamic token rather than a manually named class.

## Learning Setup

The intended direction is self-supervised or unsupervised action inference:

1. Segment objects in each frame.
2. Convert each segmented object into a point-based representation.
3. Track points across time.
4. Encode point trajectories into latent action vectors.
5. Cluster or tokenize those vectors into reusable action primitives.
6. Optionally compare discovered action tokens against known environment actions for evaluation.

## Visual Problems

The project should progress through five visual problems with increasing difficulty. Each problem should test whether object geometry and point dynamics can recover meaningful latent actions.

### 1 - Lunar Lander

#### Problem

Infer the lander's latent action from simple rendered physics.

#### Plot

A lander moves through a 2D world under gravity, using left, main, and right engine impulses. The goal is to segment the lander, track its geometric points, and recover motion tokens that align with engine effects.

LunarLander is the first controlled testbed because it provides:

- rendered video frames
- known simulator state
- known discrete actions
- simple object geometry
- interpretable physical dynamics

The simulator actions are not the main representation. They are only used as a sanity-check signal.

Plan:

1. Run the environment using a heuristic policy.
2. Save frames, observations, rewards, and simulator actions.
3. Segment the lander from each frame.
4. Extract geometric points from the lander shape.
5. Track those points through time.
6. Build latent action vectors from point dynamics.
7. Check whether discovered motion tokens align with engine actions.

### 2 - Car Racing

#### Problem

Infer vehicle control from ego-object motion and track geometry.

#### Plot

A car moves on a 2D race track with steering, acceleration, and braking. The action representation should capture vehicle translation, rotation, speed changes, and position relative to the road boundary.

### 3 - Recorded Traffic With People

#### Problem

Infer human and vehicle motion primitives from a real recorded scene.

#### Plot

People and vehicles move through a real video scene. The action representation should capture detection geometry, object trajectories, pedestrian motion, vehicle motion, and relative motion between objects.

### 4 - Random YouTube Driving Scene

#### Problem

Infer traffic actions from a separate random 3-minute YouTube driving clip.

#### Plot

The clip is a driving scene with cars and road context. This problem is harder than the local simulator and the small recorded traffic sample because it uses a longer real-world video segment with more visual variation.

Video source:

- YouTube: https://www.youtube.com/watch?v=7EovwWQIvBo
- Local clip: `external_videos/problem_4_youtube_random.mp4`

### 5 - Hard Vehicle-Crowd Interaction

#### Problem

Infer actions in the hardest separate vehicle/crowd YouTube driving clip.

#### Plot

The clip is a longer and harder real-world driving scene with dense urban activity. It tests detection and action inference under more objects, more occlusion, changing scene layout, and harder vehicle/pedestrian interactions.

Video source:

- YouTube: https://www.youtube.com/watch?v=7HaJArMDKgI
- Local clip: `external_videos/problem_5_youtube_hardest.mp4`

## Object Detection

Each visual problem has a matching detection subsection in the notebook:

### 1 - Lunar Lander Detection

YOLO is run directly on LunarLander frames. The model decides what, if anything, it recognizes from its pretrained class set; no `lander` label is provided manually.

### 2 - Car Racing Detection

YOLO is run directly on CarRacing frames. The model decides what, if anything, it recognizes from its pretrained class set; no `car` label is provided manually.

### 3 - Recorded Traffic With People Detection

YOLO detections show pedestrians and vehicles in the recorded scene. The notebook displays the annotated frame and plots people/vehicle counts over time.

### 4 - Random YouTube Driving Scene Detection

YOLO detections are shown on the independent random YouTube driving clip. The notebook displays the full local 3-minute clip, an annotated sampled frame, and people/vehicle count plots over sampled frames.

### 5 - Hard Vehicle-Crowd Interaction Detection

YOLO detections are shown on the independent hardest YouTube vehicle/crowd clip. The notebook displays the full local 3-minute clip, an annotated sampled frame, and people/vehicle count plots over sampled frames.

Each object detection subsection now renders a short annotated video clip with YOLO boxes and model-assigned class names. The plot remains as a summary over all loaded or sampled frames.

Detection now also has a lightweight tracking layer. YOLO still detects objects frame by frame, but the notebook links matching boxes over time with IoU and assigns a `track_id`. This makes it visible when a detected object persists, disappears, reappears, or gets split into a new track.

## Object Segmentation

This section uses SAM-style class-agnostic mask proposals through `FastSAM`. Unlike object detection, the segmentation step does not assign semantic classes such as `person` or `car`.

The notebook runs two segmentation experiments over the same mask proposals.

### Object Segmentation Box

This variant compresses each mask into a bounding box plus centroid. It is intentionally simpler and tests whether box-level geometry is enough for action inference.

Each box object is represented as:

- `object_id`
- bounding box derived from the mask
- area
- centroid

### Object Segmentation Blob

This variant keeps the full mask shape. It is the more precise representation for geometry-aware action inference.

Each blob object is represented as:

- `object_id`
- binary mask
- area
- centroid
- optional bounding box as a derived geometry feature

The notebook displays short segmentation videos for both variants. `Object Segmentation Box` renders boxes and centroids over a short clip. `Object Segmentation Blob` renders colored masks, mask boundaries, and centroids over a short clip. Both variants plot mask count plus total mask area over sampled frames for each visual problem.

Each segmentation subsection ends with a data representation table. Box experiments expose columns such as `frame`, `track_id`, `object_id`, `area`, `centroid_x`, `centroid_y`, `x1`, `y1`, `x2`, `y2`, `width`, and `height`. Blob experiments expose `frame`, `track_id`, `object_id`, `area`, centroid coordinates, `mask_shape`, `mask_pixels`, and a sampled list of boundary points.

Segmentation tracking is also heuristic. FastSAM proposes masks independently per frame, then the notebook links masks across frames by mask IoU. `object_id` is local to a single frame. `track_id` is the temporal identity candidate used for future trajectory and action encoding.

## Background Segmentation

Background segmentation is dynamic, not static. The goal is not only to mark background pixels, but to track reference points on the background across time.

The notebook uses Lucas-Kanade optical flow:

```text
background_point_t -> background_point_t+1
```

For each visual problem, the notebook:

1. Selects trackable image points with `cv2.goodFeaturesToTrack`.
2. Tracks them into the next frame with `cv2.calcOpticalFlowPyrLK`.
3. Estimates dominant background motion from the median point flow.
4. Marks points close to the dominant motion as `is_background`.
5. Shows a short video with motion arrows.
6. Plots point motion vectors and speed distribution.
7. Displays a data table.

Each background tracking table exposes:

- `point_id`
- `x_t`, `y_t`
- `x_t1`, `y_t1`
- `dx`, `dy`
- `speed`
- `residual_from_dominant_motion`
- `is_background`

This section is the first step toward defining action relative to the scene:

```text
relative_action = object_motion - local_background_motion
```

## Experiment Structure

Each experiment should follow the same structure:

```text
Experiment N: environment or video domain
```

Each experiment should define:

- objects to segment
- point representation
- temporal window
- latent action vector
- evaluation signal, if available

## Current Project State

The notebook now has four main experimental layers:

1. `Visual Problems`
2. `Object Detection`
3. `Object Segmentation`
4. `Background Segmentation`

The current visual problem ladder is:

1. Lunar Lander
2. Car Racing
3. Recorded traffic with people
4. Random YouTube driving scene
5. Hard vehicle-crowd interaction

YOLO is used as an external semantic detector. It works well on real-world videos, where classes such as `person`, `car`, `bus`, and `truck` exist in the pretrained model. It does not work reliably on rendered Gym environments such as Lunar Lander or Car Racing, because those images are outside the natural-image distribution of the pretrained detector.

FastSAM is used for class-agnostic segmentation. This is closer to the core project idea: first discover blobs geometrically, then assign points, trajectories, and latent action vectors.

The segmentation layer has two parallel experiments:

- `Object Segmentation Box`: compress each segmented mask into a bounding box and centroid.
- `Object Segmentation Blob`: keep the full mask shape, mask boundary, centroid, and sampled boundary points.

Each segmentation experiment ends with a data representation table. This makes the output usable for the next stages: point extraction, tracking, trajectory construction, and action encoding.

The current notebook now explicitly separates:

```text
per-frame object proposal -> temporal track_id -> geometry table -> future action vector
```

This is the first concrete bridge from visual perception to action inference. The immediate next step is to define action from tracked object geometry relative to tracked background motion.

## Notes On Oversegmentation

Oversegmentation is not necessarily a problem for this project.

If the model segments a car window instead of the whole car, that can still be useful. The window is part of the car and usually shares the same global motion as the car. In that sense, parts such as windows, doors, wheels, shadows, or body panels can act like additional views or data augmentation for the same underlying action.

Example:

```text
car moves right
window moves right
door panel moves right
wheel moves right plus local rotation
```

These fragments can later be grouped by shared motion. If several masks have highly similar trajectories, they may belong to one larger dynamic object.

Static regions are also not automatically wrong. A sidewalk, road, building, grass patch, or wall can produce a valid action representation:

```text
sidewalk -> static
road -> static or camera-motion-only
building -> static
grass -> static
```

These static objects can become useful reference frames for measuring the motion of other objects. They help distinguish object motion from camera motion.

The current working assumption is:

```text
If the video is real, the segmented blobs are valid candidates.
```

Some candidates may later be treated as:

- independent objects
- parts of larger objects
- background/reference objects
- transient artifacts

But they should not be discarded too early. The action model should learn which segments produce meaningful, stable, or reusable dynamics.

## Data Quality Warning

The main risk is not oversegmentation. The main risk is training on physically misleading video.

Real videos preserve real-world physical constraints. Even if segmentation is imperfect, the object dynamics are grounded in reality.

Non-realistic videos can poison the action model. Cartoons, superhero scenes, fantasy videos, or heavily edited clips can show impossible dynamics:

```text
a human flies without wings
a car jumps unrealistically
a robot teleports
an object changes shape without physical cause
```

If the model learns from those examples without context, it may encode impossible action priors. For example, after training on a Superman-style clip, the model could learn that a human-shaped object can fly without wings or propulsion.

For the current stage, realistic video should be preferred. Synthetic environments are still useful when their physics are known and controlled, such as Lunar Lander or Car Racing, but unrealistic visual narratives should be avoided unless they are explicitly marked as non-real or out-of-distribution.

## Key Principle

Do not begin with semantic action labels.

Begin with:

```text
objects -> points -> trajectories -> latent action tokens
```

Then evaluate whether those tokens recover meaningful behaviors.
