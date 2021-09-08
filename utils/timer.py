import time

class Timer:

    def __init__(self, name=None):
        self.name = name
        self.total_secs = 0.
    def start_timer(self):
        self.timer = time.time()
    def stop_timer(self):
        self.total_secs += time.time() - self.timer

    def __enter__(self):
        self.start_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_timer()

    def get_total_secs(self):
        return self.total_secs
    def get_total_secs_str(self):
        return f"{self.name} time: {self.total_secs}"