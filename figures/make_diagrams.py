import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_FIGS = HERE
PAGE = None          # --page <file.html>: also write a page with both figures inline
LH = 14

FILE_STYLE = """<style>
.bg { fill: #ffffff; stroke: #d0d7de; }
text { font-family: -apple-system, "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 12.5px; fill: #1f2328; }
text.t1 { font-weight: 600; }
text.lab { font-size: 11.5px; fill: #59636e; }
text.note { font-size: 11.5px; fill: #59636e; font-style: italic; }
text.ttl { font-size: 15px; font-weight: 600; }
.box { fill: #ffffff; stroke: #1f2328; stroke-width: 1.2; }
.box.wv { stroke: #0969da; stroke-width: 1.6; }
.box.gm { stroke: #bc4c00; stroke-width: 1.6; }
.dash { fill: none; stroke: #1f2328; stroke-width: 1.1; stroke-dasharray: 6 4; }
.ln { fill: none; stroke: #1f2328; stroke-width: 1.3; }
.ln.wv { stroke: #0969da; } .ln.gm { stroke: #bc4c00; } .ln.dl { stroke-dasharray: 5 4; }
.ah { fill: #1f2328; } .ah.wv { fill: #0969da; } .ah.gm { fill: #bc4c00; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #0d1117; stroke: #30363d; }
  text { fill: #e6edf3; } text.lab, text.note { fill: #9198a1; }
  .box { fill: #0d1117; stroke: #e6edf3; } .box.wv { stroke: #58a6ff; } .box.gm { stroke: #f0883e; }
  .dash, .ln { stroke: #e6edf3; } .ln.wv { stroke: #58a6ff; } .ln.gm { stroke: #f0883e; }
  .ah { fill: #e6edf3; } .ah.wv { fill: #58a6ff; } .ah.gm { fill: #f0883e; }
}
</style>"""


def esc(s):
    return html.escape(s, quote=False)


class Fig:
    def __init__(self, fid, w, h, label):
        self.fid, self.w, self.h, self.label = fid, w, h, label
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def _lines(self, x, y, w, h, lines, bold_first):
        n = len(lines)
        y0 = y + h / 2 - (n - 1) * LH / 2 + 4
        for i, t in enumerate(lines):
            c = "t1" if (i == 0 and bold_first and n > 1) else ""
            self.add('<text class="%s" x="%g" y="%g" text-anchor="middle">%s</text>'
                     % (c, x + w / 2, y0 + i * LH, esc(t)))

    def box(self, x, y, w, h, lines, cls=""):
        self.add('<rect class="box %s" x="%g" y="%g" width="%g" height="%g" rx="2"/>'
                 % (cls, x, y, w, h))
        self._lines(x, y, w, h, lines, True)

    def dbox(self, x, y, w, h, lines):
        self.add('<rect class="dash" x="%g" y="%g" width="%g" height="%g"/>' % (x, y, w, h))
        self._lines(x, y, w, h, lines, False)

    def arrow(self, d, cls="", dashed=False):
        self.add('<path class="ln %s%s" d="%s" marker-end="url(#%s-%s)"/>'
                 % (cls, " dl" if dashed else "", d, self.fid, cls or "k"))

    def line(self, d, cls="", dashed=False):
        self.add('<path class="ln %s%s" d="%s"/>' % (cls, " dl" if dashed else "", d))

    def text(self, x, y, s, anchor="middle", cls="lab"):
        self.add('<text class="%s" x="%g" y="%g" text-anchor="%s">%s</text>'
                 % (cls, x, y, anchor, esc(s)))

    def sum(self, cx, cy, r=17):
        self.add('<circle class="box" cx="%g" cy="%g" r="%g"/>' % (cx, cy, r))
        self.add('<path class="ln" d="M%g,%g H%g M%g,%g V%g"/>'
                 % (cx - 8, cy, cx + 8, cx, cy - 8, cy + 8))

    def _defs(self):
        return "<defs>%s</defs>" % "".join(
            '<marker id="%s-%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse"><path class="ah %s" d="M0,0 L10,5 '
            'L0,10 z"/></marker>' % (self.fid, k, "" if k == "k" else k)
            for k in ("k", "wv", "gm"))

    def inline(self):
        return ('<svg class="dg" viewBox="0 0 %d %d" role="img" aria-label="%s">%s%s</svg>'
                % (self.w, self.h, esc(self.label), self._defs(), "".join(self.parts)))

    def standalone(self, title, light_only=False):
        style = FILE_STYLE.split("@media")[0] + "</style>" if light_only else FILE_STYLE
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" '
                'height="%d" role="img" aria-label="%s"><title>%s</title>%s%s'
                '<rect class="bg" x="0.5" y="0.5" width="%d" height="%d" rx="6"/>%s</svg>\n'
                % (self.w, self.h, self.w, self.h, esc(self.label), esc(title), style,
                   self._defs(), self.w - 1, self.h - 1, "".join(self.parts)))


def overview():
    f = Fig("ov", 1180, 400, "Auralis engine overview: the scene feeds a wave-based FDTD solver "
            "and a geometrical-acoustics solver, joined by a zero-phase complementary crossover at "
            "the transition frequency; the hybrid impulse responses give the room-acoustic "
            "parameters and the auralisation.")
    ins = [("Geometry", "triangles, layers"), ("Materials", "α 63 Hz–8 kHz, scattering"),
           ("Sources", "directivity, trajectory"), ("Receivers", "position, head (HRTF)"),
           ("Air", "T, RH → c, ISO 9613-1")]
    for i, (a, b) in enumerate(ins):
        y = 40 + i * 66
        f.box(20, y, 162, 50, [a, b])
        f.line("M182,%g H214" % (y + 25))
    f.line("M214,65 V329")
    f.box(250, 30, 230, 118, ["Wave-based solver (FDTD)", "voxel grid, dx = c / (13.5 fc)",
                              "conformal cut-cell boundaries", "passive wall filters (impedance)",
                              "4th-order stencil on 1–4 GPUs", "free-field calibration"], "wv")
    f.box(250, 232, 230, 118, ["Geometrical acoustics", "image sources (specular part)",
                               "UTD edge diffraction", "stochastic ray tracing",
                               "size-dependent scattering", "→ events, energy echogram"], "gm")
    f.arrow("M214,89 H250", "wv")
    f.arrow("M214,291 H250", "gm")
    f.text(232, 82, "scene")
    f.text(232, 284, "scene")
    f.box(560, 256, 160, 70, ["Synthesis", "binaural (HRTF)", "Ambisonics, order 1–5"], "gm")
    f.arrow("M480,291 H560", "gm")
    f.arrow("M480,89 H760", "wv")
    f.text(620, 82, "pressure at the probes")
    f.box(760, 69, 110, 40, ["low-pass fc"], "wv")
    f.box(760, 271, 110, 40, ["high-pass fc"], "gm")
    f.arrow("M720,291 H760", "gm")
    f.dbox(510, 168, 230, 48, ["Crossover frequency fc", "auto: cell budget + Schroeder"])
    f.arrow("M740,184 H815 V109", dashed=True)
    f.arrow("M740,200 H815 V271", dashed=True)
    f.arrow("M510,192 H365 V148", dashed=True)
    f.sum(920, 190)
    f.arrow("M870,89 H920 V173")
    f.arrow("M870,291 H920 V207")
    f.text(912, 150, "zero-phase", anchor="end")
    f.text(912, 162, "crossover", anchor="end")
    f.box(980, 160, 180, 60, ["Hybrid impulse responses", "binaural + Ambisonics"])
    f.arrow("M937,190 H980")
    f.box(980, 40, 180, 60, ["Auralisation", "static or moving sources"])
    f.arrow("M1070,160 V100")
    f.box(980, 280, 180, 60, ["Room-acoustic parameters", "ISO 3382-1, G, SPL, STI"])
    f.arrow("M1070,220 V280")
    f.text(1077, 254, "omni IR (W)", anchor="start")
    f.text(590, 385, "Blue: wave band, below fc.  Orange: geometrical band.  Solid: signal.  "
                     "Dashed: control.", cls="note")
    return f


def moving():
    f = Fig("mv", 1180, 470, "Moving-source auralisation: the direct path runs through a variable "
            "delay, an interpolated read that produces the Doppler shift and HRTFs cross-faded "
            "from hop to hop; early reflections, late reverberation and the wave band are added, "
            "then all sources and a diffuse background.")
    f.dbox(110, 18, 650, 40, ["Trajectories of the sources and the listener (positions, head orientation)"])
    f.text(20, 128, "anechoic", anchor="start", cls="t1")
    f.text(20, 142, "signal", anchor="start", cls="t1")
    f.line("M20,152 H60")
    f.arrow("M60,152 H110")
    f.box(110, 125, 150, 56, ["Variable delay", "and 1/r"])
    f.arrow("M185,58 V125", dashed=True)
    f.text(193, 82, "distance, retarded", anchor="start")
    f.text(193, 96, "emission time", anchor="start")
    f.box(300, 125, 140, 56, ["Interpolation", "(Doppler)"])
    f.arrow("M260,153 H300")
    f.box(490, 110, 130, 38, ["HRTF of hop k"])
    f.box(490, 160, 130, 38, ["HRTF of hop k − 1"])
    f.line("M440,153 H465 V129")
    f.line("M465,153 V179")
    f.arrow("M465,129 H490")
    f.arrow("M465,179 H490")
    f.arrow("M555,58 V110", dashed=True)
    f.text(563, 82, "direction, head", anchor="start")
    f.text(563, 96, "orientation, 7 bands", anchor="start")
    f.box(660, 110, 100, 88, ["Crossfade", "2048-sample", "windows"])
    for y in (124, 134, 174, 184):
        f.arrow("M620,%g H660" % y)
    f.line("M60,152 V400")
    rows = [(235, ["⊗ early reflections of the current hop",
                   "image sources + UTD, recomputed every hop (43 ms at 48 kHz)"], "", "per hop"),
            (305, ["⊗ late reverberation at key frames",
                   "ray tracing, one key frame every 0.75 s"], "", "between keys"),
            (375, ["⊗ wave band at key frames (FDTD)",
                   "one key every λ/4, by reciprocity"], "wv", "between keys")]
    for y, lines, cls, fl in rows:
        f.arrow("M60,%g H110" % (y + 25))
        f.box(110, y, 510, 50, lines, cls)
        f.arrow("M620,%g H660" % (y + 25), cls)
        f.box(660, y, 100, 50, ["Crossfade", fl], cls)
        f.line("M760,%g H800" % (y + 25))
    f.line("M760,154 H800")
    f.line("M800,154 V400")
    f.sum(840, 300)
    f.arrow("M800,300 H823")
    f.box(900, 272, 170, 56, ["All sources", "gains, speech at 60 dB(A)"])
    f.arrow("M857,300 H900")
    f.box(900, 352, 170, 66, ["Diffuse background", "octave levels or a", "recording"])
    f.arrow("M985,328 V352")
    f.arrow("M1070,378 H1150")
    f.arrow("M1070,392 H1150")
    f.text(1110, 368, "binaural")
    f.text(1110, 410, "signal")
    f.text(985, 445, "diffuse-field interaural coherence; normalisation", cls="note")
    return f


FIGS = [("engine_overview", "Auralis engine overview", overview(),
         "Two solvers share the spectrum. Below the transition frequency fc, the wave-based "
         "solver integrates the wave equation on a voxel grid; above it, geometrical acoustics "
         "follows the reflections. Their responses go through complementary low-pass and "
         "high-pass filters and are summed. G and SPL take the energy of the low-passed wave band "
         "plus the share of the echogram that the high-pass lets through."),
        ("auralisation_moving", "Moving-source auralisation", moving(),
         "The direct path follows the classic chain: variable delay and 1/r attenuation, an "
         "interpolated read of the signal (which produces the Doppler shift), the HRTF of the "
         "current direction, and a crossfade from one hop to the next. Auralis adds early "
         "reflections recomputed at every hop, late reverberation and the wave band computed at "
         "key frames and cross-faded, and the diffuse background of the street or the room.")]


PAGE_TMPL = """<title>Auralis engine diagrams</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@400;500&display=swap">
<style>
/* layout: two signal-flow plates as they will sit in the repository README */
:root {
  --paper: #f6f8fa; --plate: #ffffff; --ink: #1f2328; --muted: #59636e;
  --rule: #d0d7de; --wave: #0969da; --geo: #bc4c00;
  --sans: "IBM Plex Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  --label: "IBM Plex Sans Condensed", "Arial Narrow", Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #010409; --plate: #0d1117; --ink: #e6edf3; --muted: #9198a1;
  --rule: #30363d; --wave: #58a6ff; --geo: #f0883e; color-scheme: dark } }
:root[data-theme="dark"] {
  --paper: #010409; --plate: #0d1117; --ink: #e6edf3; --muted: #9198a1;
  --rule: #30363d; --wave: #58a6ff; --geo: #f0883e; color-scheme: dark }
body { background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 16px;
  line-height: 1.55; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 36px 56px; }
main { max-width: 1200px; margin: 0 auto; display: grid; gap: 40px; }
header { display: grid; gap: 8px; max-width: 760px; }
h1 { font-size: 1.9rem; font-weight: 600; margin: 0; text-wrap: balance; }
header p { margin: 0; color: var(--muted); }
.fig { display: grid; gap: 10px; }
.fig h2 { font-size: 1.25rem; font-weight: 600; margin: 0; }
.scroll { overflow-x: auto; background: var(--plate); border: 1px solid var(--rule);
  border-radius: 6px; padding: 14px 10px; }
.dg { display: block; width: 100%; min-width: 880px; height: auto; }
.dg text { font-family: var(--label); font-size: 12.5px; fill: var(--ink); }
.dg text.t1 { font-weight: 500; }
.dg text.lab, .dg text.note { font-size: 11.5px; fill: var(--muted); }
.dg text.note { font-style: italic; }
.dg .box { fill: var(--plate); stroke: var(--ink); stroke-width: 1.2; }
.dg .box.wv { stroke: var(--wave); stroke-width: 1.6; }
.dg .box.gm { stroke: var(--geo); stroke-width: 1.6; }
.dg .dash { fill: none; stroke: var(--ink); stroke-width: 1.1; stroke-dasharray: 6 4; }
.dg .ln { fill: none; stroke: var(--ink); stroke-width: 1.3; }
.dg .ln.wv { stroke: var(--wave); } .dg .ln.gm { stroke: var(--geo); }
.dg .ln.dl { stroke-dasharray: 5 4; }
.dg .ah { fill: var(--ink); } .dg .ah.wv { fill: var(--wave); } .dg .ah.gm { fill: var(--geo); }
.cap { margin: 0; max-width: 72ch; }
code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: .9em; }
footer { color: var(--muted); font-size: 14px; max-width: 72ch; }
</style>
<main>
<header>
  <h1>Auralis engine diagrams</h1>
  <p>The two figures of the repository README, in English: the engine as a whole, and the
  signal chain that renders moving sources.</p>
</header>
%FIGS%
<footer>Repository files: <code>figures/engine_overview.svg</code> and
<code>figures/auralisation_moving.svg</code> (they follow the reader's light or dark theme on
GitHub).</footer>
</main>
"""


def main():
    """Write figures/engine_overview.svg and figures/auralisation_moving.svg (README, light or
    dark with the reader's theme), their light copies in docs/figures/ (Read the Docs); with
    --page <file.html>, also a page holding both figures inline."""
    page = sys.argv[sys.argv.index("--page") + 1] if "--page" in sys.argv else PAGE
    blocks = []
    docs_figs = os.path.join(os.path.dirname(REPO_FIGS), "docs", "figures")
    os.makedirs(docs_figs, exist_ok=True)
    for stem, title, fig, cap in FIGS:
        with open(os.path.join(REPO_FIGS, stem + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(fig.standalone(title))
        # the Read the Docs theme is always light: a light copy for docs/
        with open(os.path.join(docs_figs, stem + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(fig.standalone(title, light_only=True))
        blocks.append('<section class="fig"><h2>%s</h2><div class="scroll">%s</div>'
                      '<p class="cap">%s</p></section>' % (esc(title), fig.inline(), esc(cap)))
    if page:
        with open(page, "w", encoding="utf-8") as fh:
            fh.write(PAGE_TMPL.replace("%FIGS%", "\n".join(blocks)))


if __name__ == "__main__":
    main()
