# TrailBlazer: Peekaboo comparison

In TrailBlazer repository, our new progress integrates all the latest Peekaboo
development into here for better and easy comparison. We clone the Peekaboo
based on the commit 6564274d5329644b51c75f4e4f6f86d56edf96a9 as one of our
package module [**TrailBlazer/Baseline/Peekaboo**](../TrailBlazer/Baseline).

To download their repository please do
```bash
# at TrailBlazer root
git clone https://github.com/microsoft/Peekaboo.git TrailBlazer/Baseline/Peekaboo
# just make sure it is at that commit
cd TrailBlazer/Baseline/Peekaboo && git checkout 6564274d5329644b51c75f4e4f6f86d56edf96a9
```

## CmdPeekaboo


For your convenience, we've configured the command as the entry point for
Peekaboo, named **CmdPeekaboo**, based on their
[src/generate.py](https://github.com/microsoft/Peekaboo/blob/main/src/generate.py).
This command now accepts our configurations, allowing you to manipulate the bbox
easily, which is hard-coded in their implementation
[here](https://github.com/microsoft/Peekaboo/blob/main/src/generate.py#L92). To
run Peekaboo, simply follow our convention:

```bash
# assume ZEROSCOPE_MODEL_ROOT is set
python bin/CmdPeekaboo.py --config config/config.yaml
```


To ensure the re-producibility, you can try to replace the following code to
allow the model in customized path at their
[src/generate.py#L74](https://github.com/microsoft/Peekaboo/blob/main/src/generate.py#L74):

```python
if args.model == "zeroscope":
    model_id = "cerspense/zeroscope_v2_576w"
    model_folder = "/your/model/root"
    model_path = os.path.join(model_folder, model_id)
    pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
        model_path, torch_dtype=torch.float, variant="fp32"
    ).to(torch_device)
```

Then run the suggested command from their page,

```bash
python src/generate.py --model zeroscope --prompt "A panda eating bamboo in a lush bamboo forest" --fg_object "panda"
# Result: src/demo/zeroscope/A_panda_eating_bamboo_in_a_lush_bamboo_forest/2_of_50_2_peekaboo.mp4
```

Then run the refurbished alternative CmdPeekaboo with the config that mimic the
default bbox in their code:

```bash
python bin/CmdPeekaboo.py --config config/Peekaboo/Peekaboo-Reproduce.yaml
# Result: /tmp/Peekaboo/Peekaboo-Reproduce.0000.mp4
```

## Visual comparison

Here we show the visual comparison between the implementation from Peekaboo, our
CmdPeekaboo for reproducibility, and our TrailBlazer method. Specifically,

The following result is produced by src/generate.py, the original Peekaboo command:

<img src="../assets/v1-Peekaboo-Repro/2_of_50_2_peekaboo.gif" width="576" height="320">

This is the reproducibility from our CmdPeekaboo, with the use of
config/Peekaboo/Peekaboo-Reproduce.yaml. Please be aware that the noticeable
difference may stem from the fact that we are employing static bounding boxes
without jittering in their implementation. Nonetheless, both results appear
similar.

<img src="../assets/v1-Peekaboo-Repro/Peekaboo-Reproduce.0000-by-Peekaboo.gif" width="576" height="320">

If you're curious, here is the result obtained from the TrailBlazer command, CmdTrailBlazer:

<img src="../assets/v1-Peekaboo-Repro/Peekaboo-Reproduce.0000-by-TrailBlazer.gif" width="576" height="320">

The corresponding masks used in the default setting is here generated from
[src/generate.py#L110](https://github.com/microsoft/Peekaboo/blob/main/src/generate.py#L110):

<img src="../assets/v1-Peekaboo-Repro/mask.png" width="576" height="320">

Sweet!
