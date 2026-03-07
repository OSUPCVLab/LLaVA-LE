"""
Gradient logging callback for tracking gradient statistics during training.
Uses gradient hooks to capture gradients as they're computed (works with DeepSpeed ZeRO).
Provides two-level statistics: group-level and parameter-level.
"""
import os
import json
import torch
from transformers import TrainerCallback
from datetime import datetime


class GradientLoggingCallback(TrainerCallback):
    """
    Callback to log gradient statistics during training.
    Works with DeepSpeed ZeRO by using gradient hooks.
    Provides hierarchical statistics: parameter groups and individual parameters.
    """
    
    def __init__(self, log_every_n_steps=10, output_dir="./gradient_logs"):
        """
        Args:
            log_every_n_steps: Log gradients every n training steps
            output_dir: Directory to save gradient logs
        """
        self.log_every_n_steps = log_every_n_steps
        self.output_dir = output_dir
        self.gradient_stats = []
        self.first_step_logged = False
        self.hooks = []
        self.current_gradients = {}
        self.current_step = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"gradient_stats_{timestamp}.json")
        
        print(f"\n{'='*60}")
        print(f"GradientLoggingCallback initialized!")
        print(f"  Log every: {log_every_n_steps} steps")
        print(f"  Output file: {self.log_file}")
        print(f"  Format: JSON array")
        print(f"  Two-level stats: group + parameter")
        print(f"{'='*60}\n")
    
    @staticmethod
    def _clean_name(name: str) -> str:
        """
        Remove PEFT wrapper prefixes so parameter names
        match the LLaVA architecture (same as params.py).
        """
        for prefix in [
            "base_model.model.",
            "model.model.",
            "model."
        ]:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    def _classify_parameter(self, name: str) -> str:
        """
        Classify parameter into groups (matches params.py logic).
        """
        clean_name = self._clean_name(name)
        
        if "lora_" in clean_name:
            return "lora_adapters"
        elif clean_name.startswith("mm_projector"):
            return "mm_projector"
        elif clean_name.startswith("vision_tower"):
            return "vision_tower"
        elif clean_name.startswith("lm_head"):
            return "lm_head"
        else:
            return "other_trainable"
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Register gradient hooks on model parameters."""
        if model is None:
            return
        
        # Only register hooks on rank 0
        if args.local_rank not in [0, -1]:
            return
        
        print("Registering gradient hooks...")
        
        def make_hook(param_name):
            def hook(grad):
                # Store gradient when it's computed
                if self.current_step % self.log_every_n_steps == 0:
                    self.current_gradients[param_name] = grad.detach().clone().cpu()
                return grad
            return hook
        
        # Register hooks on all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(make_hook(name))
                self.hooks.append(handle)
        
        print(f"Registered {len(self.hooks)} gradient hooks\n")
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.current_step = state.global_step

        if state.global_step % self.log_every_n_steps != 0:
            return

        if args.local_rank not in [0, -1]:
            return

        if self.current_gradients:
            self._log_gradients(state.global_step)
            self.current_gradients = {}

            # WRITE TO DISK EVERY LOG STEP
            with open(self.log_file, "w") as f:
                json.dump(self.gradient_stats, f, indent=2)

        # Optional lightweight checkpoint every 100 steps
        if state.global_step % 100 == 0:
            partial_path = os.path.join(self.output_dir, "gradient_partial.json")
            with open(partial_path, "w") as f:
                json.dump({
                    "steps_logged": len(self.gradient_stats),
                    "last_step": state.global_step
                }, f, indent=2)

        
    def _log_gradients(self, step):
        """Process and log the captured gradients with two-level statistics."""
        if not self.current_gradients:
            print(f"Warning: No gradients captured at step {step}")
            return
        
        stats = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "parameter_groups": {}
        }
        
        # Organize gradients by group
        param_groups = {
            "lora_adapters": {},
            "mm_projector": {},
            "vision_tower": {},
            "lm_head": {},
            "other_trainable": {}
        }
        
        # Debug on first step
        if not self.first_step_logged:
            print(f"\n=== First Step: Gradient Capture at Step {step} ===")
            print(f"Total gradients captured: {len(self.current_gradients)}")
        
        # Classify and store gradients with clean names
        for name, grad in self.current_gradients.items():
            if grad.numel() == 0:
                continue
            
            group = self._classify_parameter(name)
            clean_name = self._clean_name(name)
            param_groups[group][clean_name] = grad
            
            # Debug output
            if not self.first_step_logged and len(param_groups[group]) <= 3:
                print(f"  [{group}] {clean_name}: shape={grad.shape}, norm={torch.norm(grad).item():.6f}")
        
        # Compute two-level statistics for each group
        for group_name, params_dict in param_groups.items():
            if not params_dict:
                continue
            
            # Level 1: Individual parameter statistics
            param_stats = {}
            all_grads_list = []
            
            for param_name, grad in params_dict.items():
                grad_flat = grad.flatten()
                all_grads_list.append(grad_flat)
                
                # Compute per-parameter stats
                param_stats[param_name] = {
                    "shape": list(grad.shape),
                    "num_elements": grad.numel(),
                    "l2_norm": torch.norm(grad, p=2).item(),
                    "mean": torch.mean(grad).item(),
                    "std": torch.std(grad).item(),
                    "max": torch.max(grad).item(),
                    "min": torch.min(grad).item(),
                    "max_abs": torch.max(torch.abs(grad)).item(),
                    "zero_ratio": (grad == 0).float().mean().item(),
                }
            
            # Level 2: Group-level aggregate statistics
            all_grads = torch.cat(all_grads_list)
            
            group_aggregate = {
                "num_params": len(params_dict),
                "total_elements": len(all_grads),
                "l2_norm": torch.norm(all_grads, p=2).item(),
                "mean": torch.mean(all_grads).item(),
                "std": torch.std(all_grads).item(),
                "max": torch.max(all_grads).item(),
                "min": torch.min(all_grads).item(),
                "max_abs": torch.max(torch.abs(all_grads)).item(),
                "zero_ratio": (all_grads == 0).float().mean().item(),
                "near_zero_ratio": (torch.abs(all_grads) < 1e-7).float().mean().item(),
            }
            
            # Top 5 parameters by gradient norm
            top_params = sorted(
                [(name, param_stats[name]["l2_norm"]) for name in params_dict.keys()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            group_aggregate["top_5_by_norm"] = [
                {"param": name, "l2_norm": norm} for name, norm in top_params
            ]
            
            # Store hierarchical stats
            stats["parameter_groups"][group_name] = {
                "group_stats": group_aggregate,
                "parameters": param_stats
            }
        
        # Overall statistics across all groups
        all_param_grads = []
        total_params = 0
        for group_data in stats["parameter_groups"].values():
            for grad in param_groups[list(stats["parameter_groups"].keys())[list(stats["parameter_groups"].values()).index(group_data)]].values():
                all_param_grads.append(grad.flatten())
                total_params += 1
        
        if all_param_grads:
            all_grads_flat = torch.cat(all_param_grads)
            stats["overall"] = {
                "total_params": total_params,
                "total_elements": len(all_grads_flat),
                "l2_norm": torch.norm(all_grads_flat, p=2).item(),
                "mean": torch.mean(all_grads_flat).item(),
                "std": torch.std(all_grads_flat).item(),
                "max_abs": torch.max(torch.abs(all_grads_flat)).item(),
            }
        
        # Mark first step
        if not self.first_step_logged:
            self.first_step_logged = True
            print(f"\nParameter group counts:")
            for group_name, params in param_groups.items():
                if params:
                    print(f"  {group_name}: {len(params)} parameters")
            print("=" * 50 + "\n")
        
        # Print summary (don't save to file yet - accumulate in memory)
        if "overall" in stats:
            print(f"\n[Step {step}] Gradient Stats:")
            print(f"  Overall L2 Norm: {stats['overall']['l2_norm']:.6f}")
            
            for group_name, group_data in stats["parameter_groups"].items():
                group_stats = group_data["group_stats"]
                print(f"  {group_name}: L2={group_stats['l2_norm']:.6f}, "
                      f"mean={group_stats['mean']:.6e}, "
                      f"std={group_stats['std']:.6e}, "
                      f"params={group_stats['num_params']}")
        
        self.gradient_stats.append(stats)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training - write all accumulated stats to JSON."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        if args.local_rank not in [0, -1]:
            return
        
        # Write all gradient stats as JSON array
        if self.gradient_stats:
            with open(self.log_file, 'w') as f:
                json.dump(self.gradient_stats, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"Gradient logging complete!")
            print(f"  Total steps logged: {len(self.gradient_stats)}")
            print(f"  Output file: {self.log_file}")
            print(f"  File size: {os.path.getsize(self.log_file) / 1024:.1f} KB")
            print(f"{'='*60}\n")
        else:
            print("\nWarning: No gradient statistics were collected.")
        
        # Create summary file
        if self.gradient_stats:
            summary_file = os.path.join(self.output_dir, "gradient_summary.json")
            
            # Collect all unique parameter groups across all steps
            all_groups = set()
            for stat in self.gradient_stats:
                all_groups.update(stat.get("parameter_groups", {}).keys())
            
            summary = {
                "total_steps_logged": len(self.gradient_stats),
                "log_file": self.log_file,
                "parameter_groups": sorted(list(all_groups)),
                "steps": [stat["step"] for stat in self.gradient_stats]
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Summary saved to: {summary_file}\n")
