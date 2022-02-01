from timeit import default_timer as timer
import humanize


class Timer():
    def __init__(self, start_msg=None):
        if start_msg is not None:
            print(start_msg)

        self.start = timer()
        self.prev = self.start
        self.i = 0

    def __call__(self, info=None, fin=False):
        now = timer()
        delta = now - self.prev

        took = humanize.naturaldelta(delta)

        msg = f"[{self.i}] - {took}"

        if info is not None:
            msg += f": {info}"

        print(msg)
        self.prev = now
        self.i += 1

        if fin:
            delta = self.prev - self.start
            took = humanize.naturaldelta(delta, minimum_unit="microseconds")
            msg = f"[done] - {took}"
