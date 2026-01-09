from fast_multitalk.pipeline import MultiTalkPipeline


distill_model = "fusionx"
quant = "fp6_e3m2"
low_vram_mode = True
use_teacache = True
force_9_16 = True
i2v = MultiTalkPipeline(
    base_dir="/home/featurize/data/fast-multitalk",
    distill_model=distill_model,
    low_vram_mode=low_vram_mode,
    num_persistent_param_in_dit=0,
    quant=quant,
    cache_contexts=["人物面向镜头说话"],
)
for test_config in [
    "dsx_single1",
]:
    print(f"Testing {test_config}")
    for i in [6]:
        video = i2v.generate(
            f"tests/{test_config}.json",
            video_save_path=f"tests/outputs/{test_config}_{distill_model}_{quant}_{'teacache' if use_teacache else 'no_teacache'}_{'9-16' if force_9_16 else 'raw'}_{i}.mp4",
            sample_steps=i,
            shift=7,
            seed=42,
            use_teacache=use_teacache,
            merge_video_audio=True,
            force_9_16=force_9_16,
        )
        print(f"Sampling steps: {i}, Video saved to {video}")
