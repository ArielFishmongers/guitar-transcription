"""No-op deformers. Referenced at import time by the FretNet continuous repo
(e.g. `muda.deformers.PitchShift`) but never called on the inference path."""


class PitchShift:
    def __init__(self, *args, **kwargs):
        pass

    def transform(self, jam):
        return iter([jam])
