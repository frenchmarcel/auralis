# Listening

Auralis turns impulse responses into sound. You can listen in the browser or
render WAV files.

## The Auralizer

Pick a result, a sound and a listening mode:

- **Binaural, fixed head.** The sound is convolved with the binaural impulse
  response. This is the exact result for one head direction.
- **Head tracking.** The sound is convolved with the second-order Ambisonics
  response. When you turn your head, the field turns and a binaural decoder
  renders it. No acoustic computation runs again.

You can switch between results (A, B and C) to compare them with the same
sound.

```{figure} ../images/auralizer.jpg
:alt: The Auralizer with a top view of the seminar room and the playback controls
:width: 100%

The Auralizer for the seminar room. Click a receiver in the top view to listen
from it. Each source gets a sound and a gain. The A, B and C slots compare
simulations with the same sound.
```

### Sounds

You can use your own recordings. The app also has test signals. With a local
Odeon installation, it can use the Odeon anechoic recordings.

A good input is a dry recording, made in an anechoic room or very close to
the source. A recording that already has a room in it adds that room to the
result.

## Render to WAV

The server can render the auralisation to a WAV file. Each source can play its
own track. The credits of the recordings and of the measured head go into a
text file next to the WAV file.

### Ensembles

An ensemble places one source per player, each with its instrument's
directivity and its own track. This lets you hear an orchestra or a band from
any seat.

## Animated renders

An animated render follows the trajectories of the sources and of the
listener. It needs a receiver and at least one source with a sound.

What the render does:

- The **direct sound** goes through a variable delay line. This gives the
  Doppler effect and the right arrival time.
- The **early reflections** are computed again at every hop of 2048 samples
  (43 ms at 48 kHz).
- The **late reverberation** is computed at key frames, every 0.75 s by
  default, and cross-faded.
- With the wave solver, the **wave band** is computed at key frames too and
  cross-faded.

See {doc}`../engine/moving-sources` for the details.

```{figure} ../images/animated-render.jpg
:alt: The Auralizer of the Paris street with the animated render panel
:width: 100%

The Paris street in the Auralizer. The sources and the listener follow their
trajectories. The panel on the right starts the animated render. When it is
done, you can play it, download the WAV file and read the credits.
```

### Background noise

A render can add a diffuse background:

- **Octave levels**, for example an NC curve or your own levels,
- or a **recording**, for example the traffic of a city.

The background reaches the two ears with the coherence of a diffuse field, so
it sounds around you and not inside your head. Its level is set in dB(A) at
the ears. The other tracks are then set to the level of speech at the chosen
vocal effort before their gains apply.

### Turn taking

For a conversation, the render can let voices take turns. It finds the pauses
in each track and plays one speaker at a time.

## Headphones

All binaural output is made for headphones. Use closed or open headphones
without any spatial effect from the player or the system.
