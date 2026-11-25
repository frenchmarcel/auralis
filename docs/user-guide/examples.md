# Examples

On its first start, the app builds an **Examples** project.

| Example | What it shows | Needs |
|---|---|---|
| Concert halls | Mozart in the Musikverein, Mahler in Gothenburg, Bruckner at the Royal Festival Hall, with the materials of the Odeon models | Local Odeon data |
| BRAS rooms | The rooms of the BRAS database: coupled rooms (CR1, door open 30.4° and 4.1°), a seminar room (CR2), a chamber music hall (CR3) and an auditorium (CR4). Each has the geometry, the materials, the source and microphone positions and the air of the measurements. | Nothing |
| Paris street | Réaumur Street in Paris, from OpenStreetMap, with real recordings (cars, a bus, traffic, people talking). It comes with two finished renders: the full physics, and the geometrical solver only. | Nothing |

The concert halls and the BRAS rooms run in the background at the first start.
Each takes a few seconds to a few minutes. The Paris street runs nothing: its
renders were made on four large GPUs. Examples that need local data are
skipped when the data is missing.

```{figure} ../images/example-bras.jpg
:alt: The coupled rooms of BRAS with their sources and microphones
:width: 100%

The coupled rooms of BRAS (CR1): a reverberation chamber and a laboratory,
joined by a door, here open 30.4°.
```

The BRAS rooms keep the measured positions exactly. Their materials come from
BRAS, which fitted them to the measured reverberation time. Compare your
results with the measurements in {doc}`../validation/index`.

## Updating the examples

The examples have a version number. When a new version adds examples, the
app adds only the new ones to your existing project. It does not change or
delete what you already have. Version 7 removed the furnished living room and
the synthetic streets. A project that already has them keeps them.

## Credits

- The BRAS rooms use data of the BRAS database (TU Berlin, CC BY-SA 4.0).
- The Paris street uses OpenStreetMap data (© OpenStreetMap contributors,
  ODbL).
- The real recordings come from Freesound. They are in the public domain
  (CC0). Their authors are still credited in each render.
- Renders with the FABIAN head credit TU Berlin (CC BY 4.0).
- Furniture models and textures come from Poly Haven (CC0).
