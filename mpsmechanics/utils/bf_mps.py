from mps.load import MPS, logger


class BFMPS(MPS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        um_per_pixel = self.data.info["um_per_pixel"]
        if um_per_pixel != 0.325:
            self.data.info["um_per_pixel"] = 0.325
            logger.warning(
                f"um_per_pixel recorded as {um_per_pixel}; changing to {self.data.info['um_per_pixel']}"
            )
            self._unpack(self.data)
