import time
from stable_baselines3.common.callbacks import BaseCallback

class InfoCallback(BaseCallback):
    def __init__(
        self,
        total_steps: int,
        step_frequency: int = 50_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.step_frequency = step_frequency
    
    def _on_step(self) -> bool:
        """Will print an update to stdout for everytime a certain number
        of time steps has been made.
        """
        calls = self.n_calls
        # counter is incremented before _on_step() is called in BaseCallback
        if calls == 1:
            self.start_time = time.time()
        if calls % self.step_frequency == 0:
            progress_percent = round((calls / self.total_steps) * 100, 2)
            print(f"Training progress for this run: {progress_percent}%")
            print(f"Current time step: {calls}")
            exec_time = round(time.time() - self.start_time)
            print(f"Execution time since run start: {exec_time} seconds\n")
        
        return True

class EarlyStoppingCallback(BaseCallback):
    # TODO implement
    pass