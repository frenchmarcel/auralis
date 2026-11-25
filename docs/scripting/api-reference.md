# API reference

This page lists the main classes and functions of `auralis.engine`. It is
built from the docstrings in the code.

## Running a simulation

```{eval-rst}
.. autoclass:: auralis.engine.pipeline.BakeConfig
   :no-members:

.. autoclass:: auralis.engine.pipeline.BakeResult
   :no-members:

.. autoclass:: auralis.engine.pipeline.SourceDef
   :no-members:

.. autoclass:: auralis.engine.pipeline.ReceiverDef
   :no-members:

.. autofunction:: auralis.engine.pipeline.bake

.. autofunction:: auralis.engine.pipeline.bake_pair

.. autofunction:: auralis.engine.pipeline.precompute_low_bands

.. autofunction:: auralis.engine.pipeline.source_low_bands

.. autofunction:: auralis.engine.pipeline.estimate_max_time
```

## Scenes and materials

```{eval-rst}
.. autoclass:: auralis.engine.geometry.SceneData
   :members: from_materials

.. autofunction:: auralis.engine.geometry.make_shoebox

.. autofunction:: auralis.engine.geometry.add_box

.. autofunction:: auralis.engine.geometry.box_triangles

.. autofunction:: auralis.engine.materials.get_preset

.. autofunction:: auralis.engine.project.build_scene

.. autofunction:: auralis.engine.project.build_config
```

## Wave solver

```{eval-rst}
.. autofunction:: auralis.engine.fdtd.run_low_band

.. autofunction:: auralis.engine.fdtd.voxelize

.. autofunction:: auralis.engine.fdtd.design_crossover

.. autofunction:: auralis.engine.fdtd.fuse_low_high

.. autofunction:: auralis.engine.fdtd.surface_taps
```

## Moving sources

```{eval-rst}
.. autoclass:: auralis.engine.motion.Trajectory
   :no-members:

.. autoclass:: auralis.engine.motion.MovingSource
   :no-members:

.. autoclass:: auralis.engine.motion.MotionConfig
   :no-members:

.. autofunction:: auralis.engine.motion.render_moving

.. autofunction:: auralis.engine.motion.render_scene_moving
```

## Parameters

```{eval-rst}
.. autofunction:: auralis.engine.rirmetrics.band_metrics

.. autofunction:: auralis.engine.metrics.compute_metrics

.. autofunction:: auralis.engine.metrics.format_report
```

## Binaural and Ambisonics

```{eval-rst}
.. autofunction:: auralis.engine.ambidecode.decoder_filters

.. autofunction:: auralis.engine.ambidecode.decode

.. autofunction:: auralis.engine.ambidecode.rotation_matrix

.. autofunction:: auralis.engine.ir_synthesis.sh_sn3d
```

## Directivity

```{eval-rst}
.. autoclass:: auralis.engine.directivity.Directivity
   :no-members:

.. autofunction:: auralis.engine.directivity.get_directivity
```

## Audio files

```{eval-rst}
.. autofunction:: auralis.engine.wavio.read_wav

.. autofunction:: auralis.engine.wavio.write_wav

.. autofunction:: auralis.engine.convolve.fft_convolve

.. autofunction:: auralis.engine.convolve.make_test_signal
```
