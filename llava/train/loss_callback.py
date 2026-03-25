import json
from transformers import TrainerCallback

class LossJSONLogger(TrainerCallback):
    def __init__(self, output_path="loss_log.json"):
        self.output_path = output_path
        self.data = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            entry = {
                "step": state.global_step,
                "loss": float(logs["loss"]),
                "learning_rate": float(logs.get("learning_rate", 0.0))
            }
            self.data.append(entry)

            # write incrementally so it survives crashes
            with open(self.output_path, "w") as f:
                json.dump(self.data, f, indent=2)
