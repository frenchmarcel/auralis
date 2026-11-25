import argparse
import io
import json
import os
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright

CHROME_ARGS = ["--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
               "--ignore-gpu-blocklist"]
SIMS = {  # key: (geometry name prefix, simulation name prefix)
    "cr1": ("Coupled rooms, door open 30.4°", "BRAS measurement positions"),
    "cr2": ("Seminar room (BRAS CR2)", "BRAS measurement positions"),
    "cr3": ("Chamber music hall (BRAS CR3)", "BRAS measurement positions"),
    "cr4": ("Auditorium (BRAS CR4)", "BRAS measurement positions"),
    "paris": ("Réaumur Street", "Walk to Sébastopol Boulevard"),
}
PNG_SHOTS = {"app-home", "results-report"}


def find_simulations(url):
    """Map each key of SIMS to the id of the first simulation whose geometry and simulation
    names start with its prefixes."""
    with urllib.request.urlopen(url + "/api/projects") as r:
        projects = json.load(r)
    found = {}
    for p in projects:
        for s in p["spaces"]:
            for g in s["geometries"]:
                for m in g["simulations"]:
                    for key, (gname, mname) in SIMS.items():
                        if key not in found and g["name"].startswith(gname) and \
                                m["name"].startswith(mname):
                            found[key] = m["id"]
    missing = sorted(set(SIMS) - set(found))
    if missing:
        raise SystemExit("simulations not found in the store: %s" % ", ".join(missing))
    return found


class Shooter:
    """Drives one browser page through the app and saves compressed screenshots."""

    def __init__(self, page, url, sims, out):
        self.pg, self.url, self.sims, self.out = page, url, sims, out

    def go(self, route, key=None, wait=9000):
        self.pg.goto(self.url + ("/#/%s/%s" % (route, self.sims[key]) if key else "/#/"))
        self.pg.wait_for_timeout(wait)

    def button(self, text, exact=True):
        self.pg.get_by_role("button", name=text, exact=exact).first.click()
        self.pg.wait_for_timeout(800)

    def tab(self, text):
        self.pg.locator("button", has_text=text).first.click()
        self.pg.wait_for_timeout(1200)

    def view(self, text, wait=6000):
        self.pg.get_by_role("radio", name=text, exact=True).first.click()
        self.pg.wait_for_timeout(wait)

    def zoom(self, x, y, steps, dy=-240):
        self.pg.mouse.move(x, y)
        for _ in range(steps):
            self.pg.mouse.wheel(0, dy)
            self.pg.wait_for_timeout(120)
        self.pg.wait_for_timeout(1500)

    def pan(self, x, y, dx, dy, steps=10):
        self.pg.mouse.move(x, y)
        self.pg.mouse.down(button="right")
        for k in range(1, steps + 1):
            self.pg.mouse.move(x + dx * k / steps, y + dy * k / steps)
            self.pg.wait_for_timeout(40)
        self.pg.mouse.up(button="right")
        self.pg.wait_for_timeout(800)

    def shot(self, name):
        img = Image.open(io.BytesIO(self.pg.screenshot())).convert("RGB")
        if name in PNG_SHOTS:
            path = os.path.join(self.out, name + ".png")
            img.save(path, optimize=True)
        else:
            path = os.path.join(self.out, name + ".jpg")
            img.save(path, quality=90, subsampling=0, optimize=True)
        print("%-24s %4d kB" % (name, os.path.getsize(path) // 1024))


def capture(s):
    """Take every screenshot used in the documentation."""
    s.go("", wait=3000)
    s.shot("app-home")

    s.go("editor", "paris", 14000)
    s.tab("Sources")
    s.shot("example-paris")

    s.go("editor", "cr4", 12000)
    s.button("Iso")
    s.shot("editor-bras")

    s.go("editor", "cr3", 12000)
    s.button("Iso")
    s.tab("Sources")
    s.shot("sources-bras")

    s.go("editor", "cr1", 12000)
    s.button("Iso")
    s.shot("example-bras")

    s.go("editor", "cr2")
    s.button("Iso")
    s.zoom(740, 560, 3)
    s.shot("materials-tab")
    s.pg.locator("text=mat_CR2_plaster").first.click()
    s.pg.wait_for_timeout(600)
    s.button("Assign…", exact=False)
    s.pg.wait_for_timeout(2000)
    s.shot("material-library")
    s.pg.locator("button[aria-label='Close library']").first.click()
    s.pg.wait_for_timeout(800)
    s.tab("Settings")
    s.shot("settings-tab")
    s.pg.locator("button[aria-haspopup=menu]").first.click()
    s.pg.wait_for_timeout(500)
    s.pg.locator(".menu-list button", has_text="Geometry check").first.click()
    s.pg.wait_for_timeout(3000)
    s.shot("model-check")

    s.go("results", "cr2")
    s.shot("results-model")
    s.view("Report")
    s.shot("results-report")
    s.view("Grid maps")
    s.shot("results-grid-map")
    s.view("Reflections")
    s.shot("results-reflections")

    s.go("auralizer", "cr2", 7000)
    s.shot("auralizer")
    s.go("auralizer", "paris")
    s.shot("animated-render")


def main():
    ap = argparse.ArgumentParser(
        description="Take the web app screenshots of the documentation. Needs a running "
                    "test server whose store holds the example project, with the BRAS CR2 "
                    "example run in the Default mode with the FABIAN head and a receiver grid. "
                    "Never point it at your own projects: it opens dialogs but saves nothing "
                    "on purpose.")
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "images"))
    ap.add_argument("--chrome", default="/usr/bin/google-chrome")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sims = find_simulations(a.url)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome, headless=False, args=CHROME_ARGS)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.25)
        capture(Shooter(page, a.url, sims, a.out))
        browser.close()


if __name__ == "__main__":
    main()
