# TrailBlazer: Commands

Here we illustrate how to make your own config under the current implementation.
Please use the following commands to run our TrailBlazer. The config structure
is detailed [here](Config.md)

## bin/CmdTrailBlazer.py

```bash
python bin/CmdTrailBlazer.py --config ${YOUR_CONFIG_FILEPATH} -mr ${YOUR_MODEL_ROOT}
e.g., python bin/CmdTrailBlazer.py --config config/PerspBR2TL-Tiger.yaml -mr ${YOUR_MODEL_ROOT}
```

If you feel the flag -mr(--model-root) is combersome, you can create a shell
environment variable named ZEROSCOPE_MODEL_ROOT so that flag can be ignored.
It's the folder path that contains cerspense/zeroscope_v2_576w

```bash
export ZEROSCOPE_MODEL_ROOT=/your/huggingface/model/root/
# Note: then we expect the zeroscope model is here /your/huggingface/model/root/cerspense/zeroscope_v2_576w/
```

Then it above command now becomes

```bash
python bin/CmdTrailBlazer.py --config ${YOUR_CONFIG_FILEPATH}
e.g., python bin/CmdTrailBlazer.py --config config/Main/PerspBR2TL-Tiger.yaml
```

For your convenience, you could run the following command to get the results
from all config yamls in the given folder:

```bash
python bin/CmdTrailBlazer.py --config ${YOUR_CONFIG_FOLDER}
# say, you want to execute all yaml files in the provided config folder
e.g., python bin/CmdTrailBlazer.py --config config/Main/
```

## bin/CmdTrailBlazerMulti.py

As outlined in the paper, the individual objects are initially generated
independently before being integrated. For example, if one wishes to guide
both a dog and a cat based on the prompt "a dog and a cat running in the park,"
each needs to be processed separately using our **CmdTrailBlazer** command:

```bash
python bin/CmdTrailBlazer.py --config config/Multi/MultiSubject-Dog.yaml
python bin/CmdTrailBlazer.py --config config/Multi/MultiSubject-Cat.yaml
```

Upon execution, you will receive the reconstructed video as usual. Additionally,
a .pt file will be generated containing the bounding box at each frame and the
latent vectors at each step. For example:

```bash
/tmp/TrailBlazer/MultiSubject-Dog.0000.pt
/tmp/TrailBlazer/MultiSubject-Cat.0000.pt
```

After placing these .pt files in the "subjects" key within the configuration,
proceed to execute the distinct command **CmdTrailBlazerMulti** to obtain the final
result.

```bash
python bin/CmdTrailBlazerMulti.py --config config/Multi/MultiSubjects.yaml
```

## bin/CmdPeekaboo.py

We've managed the entry point of Peekaboo within our executable script
**CmdPeekaboo**, enhancing compatibility with our configuration file in their
implementation. Please checkout the page [Peekaboo.md](Peekaboo.md) for more
information about the reproducibility, and the [Config.md](Config.md) for the
configuration design about Peekaboo.

```bash
python bin/CmdPeekaboo.py --config config/config.yaml
```

We notice that the full precision is used in Peekaboo implementation (e.g.,
torch.float (See
[src/generate.py#L76](https://github.com/microsoft/Peekaboo/blob/main/src/generate.py#L76)))
that goes beyond our VRAM to run the batch execution (e.g., folder path in
--config flag). If you want to run all config under specific folder. Here is an
alternative way:

```bash
for f in config/*.yaml; do python bin/CmdPeekaboo.py --config $f; done
```

Get some coffee after executing it :)
