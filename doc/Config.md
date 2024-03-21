# TrailBlazer: Config

## Single object

The following example pseudo config covers the needs to run our TrailBlazer for
the single object.

```yaml
keyframe:
- bbox_ratios:
  - 0.5
  - 0.5
  - 1.0
  - 1.0
  frame: 0
  prompt: A tiger walking alone down the street
- bbox_ratios:
  - 0.0
  - 0.0
  - 0.1
  - 0.1
  frame: 24
  prompt: A tiger walking alone down the street
seed: 123451232531
token_inds:
- 2
width: 576
height: 320
num_inference_steps:
trailblazer:
  num_dd_spatial_steps: 5
  num_dd_temporal_steps: 5
  spatial_strengthen_scale: 0.15
  spatial_weaken_scale: 0.001
  temp_strengthen_scale: 0.15
  temp_weaken_scale: 0.001
  trailing_length: 13
text2vidzero:
  motion_field_strength_x: -8
  motion_field_strength_y: 0
peekaboo:
  frozen_steps: 2
```

Some requirements:

- At least two keyframes are required for the initial and the end frame. The
  initial frame index must be 0.

- It's recommend to set the end frame at 24 as this is how ZeroScope model was
  trained (See [here](https://zeroscope.replicate.dev/)). In this example, the
  second keyframe is set as 24.

- Each keyframe contains bbox_ratios, frame, and prompt. The consistency between
  each component should be maintained conceptually by user.

- In our experience, the trailing_length is a parameter that needs frequent
  adjustment for optimal results.

- The tuple of the floats in bbox is the left, top, right, and bottom of the
  boundary relative to the normalized image space between 0 and 1. The bbox
  should be reasonably specified. E.g., b_left < b_right, b_top < b_bottom in
  OpenCV style. (e.g., 0.0,0.0,0.5,0.5 is the second quadrant)

- There are three sections in the config calling **trailblazer**,
  **text2vidzero**, and **peekaboo**. The arguments under each category are the
  hyper-parameters of each method. For text2vidzero please checkout the external
  [link](https://huggingface.co/docs/diffusers/en/api/pipelines/text_to_video_zero).

- For Peekaboo, as the method used for main comparison in our method. The key
  frozen_steps is the only one hyper-parameter in the
  [implementation](https://github.com/microsoft/Peekaboo/blob/main/src/generate.py#L30).
  Please refer to [link](Peekaboo.md) for more information.


## Multiple object

The multiple object generation config needs

```yaml
multisubs:
  seed: 12345611
  num_integration_steps: 20
  prompt: a white cat and a yellow dog running in the botanic garden
  subjects:
    - /tmp/TrailBlazer/MultiSubject-Dog.0000.pt
    - /tmp/TrailBlazer/MultiSubject-Cat.0000.pt
```

The **num_integration_steps** key is the number of steps used for integrating
the latents between the subjects listed under **subjets** key, obtained from the
single object generation in TrailBlazer.
