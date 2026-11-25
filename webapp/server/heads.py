"""Listener heads of the binaural rendering: the parametric model, or a measured set."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADS = {
    "model": {"name": "Spherical head (model)", "path": None, "credit": None},
    "fabian": {"name": "FABIAN (measured)",
               "path": os.path.join(REPO, "data_hrtf", "fabian", "fabian_measured_hato0.npz"),
               "credit": "FABIAN HRTF database, TU Berlin (Brinkmann et al., JAES 2017), "
                         "CC BY 4.0"},
}


def hrtf_path(head):
    """Engine hrtf_path of a head id ('' for the model); raises if unknown or missing."""
    h = HEADS.get(head or "model")
    if h is None:
        raise ValueError("unknown head '%s' (one of %s)" % (head, ", ".join(HEADS)))
    if h["path"] and not os.path.exists(h["path"]):
        raise ValueError("the %s head is not installed: %s (see data_hrtf/README.md)"
                         % (h["name"], h["path"]))
    return h["path"] or ""


def listing():
    """[{id, name, available, credit}] for the interface."""
    return [{"id": k, "name": h["name"], "credit": h["credit"],
             "available": not h["path"] or os.path.exists(h["path"])} for k, h in HEADS.items()]
