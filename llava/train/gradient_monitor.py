"""
Gradient monitoring callback for tracking gradient magnitudes during training.
Works with DeepSpeed ZeRO-3 and LoRA adapters.
"""

import torch
import numpy as np
from transformers import TrainerCallback
from collections import defaultdict
import json
import os
from pathlib import Path


class GradientMonitorCallback(TrainerCallback):
    """
    Callback to monitor and log gradient statistics during training.
    Compatible with DeepSpeed ZeRO-3 and LoRA fine-tuning.
    """
    
    def __init__(
        self, 
        log_interval=10,
        output_dir="./gradient_logs",
        log_to_wandb=True,
        track_lora_only=True,
        percentiles=(10, 25, 50, 75, 90)
    ):
        """
        Args:
            log_interval: Log gradient stats every N steps
            output_dir: Directory to save gradient logs
            log_to_wandb: Whether to log to wandb
            track_lora_only: If True, only track LoRA parameters (recommended for memory)
            percentiles: Percentile values to compute for gradient distribution
        """
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_to_wandb = log_to_wandb
        self.track_lora_only = track_lora_only
        self.percentiles = percentiles
        
        # Storage for gradient statistics
        self.gradient_stats = []
        self.grad_log_file = self.output_dir / "gradient_stats.jsonl"
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize gradient monitoring at the start of training."""
        print(f"Gradient monitoring enabled - logging every {self.log_interval} steps")
        print(f"Gradient logs will be saved to: {self.grad_log_file}")
        print(f"Tracking {'LoRA parameters only' if self.track_lora_only else 'all parameters'}")
        
    def _collect_gradient_stats(self, model, step):
        """
        Collect gradient statistics from model parameters.
        Handles DeepSpeed ZeRO-3 parameter gathering.
        """
        grad_stats = {
            'step': step,
            'layer_stats': {},
            'global_stats': {}
        }
        
        all_grads = []
        layer_grads = defaultdict(list)
        
        # Iterate through named parameters
        for name, param in model.named_parameters():
            # Skip if tracking LoRA only and this isn't a LoRA parameter
            if self.track_lora_only and 'lora_' not in name:
                continue
                
            # Skip if no gradient
            if param.grad is None:
                continue
            
            try:
                # Handle DeepSpeed ZeRO-3 partitioned parameters
                if hasattr(param, 'ds_id'):
                    from deepspeed import zero
                    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                    
                    if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                        continue
                        
                    # Gather parameter gradients from all ranks
                    with zero.GatheredParameters([param], modifier_rank=0):
                        if param.grad is not None:
                            grad_data = param.grad.detach().float()
                        else:
                            continue
                else:
                    grad_data = param.grad.detach().float()
                
                # Compute gradient statistics
                grad_norm = torch.norm(grad_data).item()
                grad_mean = grad_data.mean().item()
                grad_std = grad_data.std().item()
                grad_max = grad_data.abs().max().item()
                
                # Flatten for percentile computation
                grad_flat = grad_data.flatten().cpu().numpy()
                
                # Determine layer group (e.g., lora_A, lora_B, base model)
                if 'lora_A' in name:
                    layer_group = 'lora_A'
                elif 'lora_B' in name:
                    layer_group = 'lora_B'
                elif 'mm_projector' in name:
                    layer_group = 'mm_projector'
                else:
                    layer_group = 'base_model'
                
                # Store layer-specific stats
                layer_grads[layer_group].append(grad_flat)
                
                # Collect all gradients for global stats
                all_grads.append(grad_flat)
                
                # Store per-parameter stats (limited to avoid memory issues)
                if len(grad_stats['layer_stats']) < 20:  # Limit detailed logging
                    grad_stats['layer_stats'][name] = {
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std,
                        'max': grad_max,
                    }
                    
            except Exception as e:
                # Skip parameters that cause issues (e.g., not on this rank)
                continue
        
        # Compute global statistics
        if all_grads:
            all_grads_concat = np.concatenate(all_grads)
            
            grad_stats['global_stats'] = {
                'mean': float(np.mean(all_grads_concat)),
                'std': float(np.std(all_grads_concat)),
                'norm': float(np.linalg.norm(all_grads_concat)),
                'max': float(np.max(np.abs(all_grads_concat))),
                'min': float(np.min(np.abs(all_grads_concat))),
                'num_params': len(all_grads_concat),
                'percentiles': {}
            }
            
            # Compute percentiles
            for p in self.percentiles:
                grad_stats['global_stats']['percentiles'][f'p{p}'] = float(
                    np.percentile(np.abs(all_grads_concat), p)
                )
            
            # Check for vanishing/exploding gradients
            grad_stats['global_stats']['vanishing'] = grad_stats['global_stats']['max'] < 1e-7
            grad_stats['global_stats']['exploding'] = grad_stats['global_stats']['max'] > 100.0
        
        # Compute per-layer-group statistics
        grad_stats['layer_group_stats'] = {}
        for group, grads in layer_grads.items():
            if grads:
                grads_concat = np.concatenate(grads)
                grad_stats['layer_group_stats'][group] = {
                    'mean': float(np.mean(grads_concat)),
                    'std': float(np.std(grads_concat)),
                    'norm': float(np.linalg.norm(grads_concat)),
                    'max': float(np.max(np.abs(grads_concat))),
                }
        
        return grad_stats
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Collect and log gradient statistics after each training step."""
        # Only log at specified intervals
        if state.global_step % self.log_interval != 0:
            return
        
        # Collect gradient statistics
        try:
            grad_stats = self._collect_gradient_stats(model, state.global_step)
            
            # Save to file (JSONL format for easy streaming)
            with open(self.grad_log_file, 'a') as f:
                json.dump(grad_stats, f)
                f.write('\n')
            
            # Print summary to console
            if grad_stats['global_stats']:
                gs = grad_stats['global_stats']
                print(f"\n{'='*60}")
                print(f"Gradient Stats @ Step {state.global_step}")
                print(f"{'='*60}")
                print(f"  Global Norm:  {gs['norm']:.6e}")
                print(f"  Mean (abs):   {abs(gs['mean']):.6e}")
                print(f"  Max (abs):    {gs['max']:.6e}")
                print(f"  Std:          {gs['std']:.6e}")
                
                if gs.get('vanishing'):
                    print(f"  ⚠️  WARNING: Vanishing gradients detected!")
                if gs.get('exploding'):
                    print(f"  ⚠️  WARNING: Exploding gradients detected!")
                
                # Print layer group stats
                if grad_stats.get('layer_group_stats'):
                    print(f"\n  Layer Group Statistics:")
                    for group, stats in grad_stats['layer_group_stats'].items():
                        print(f"    {group:20s} - Norm: {stats['norm']:.6e}, Max: {stats['max']:.6e}")
                
                print(f"{'='*60}\n")
            
            # Log to wandb if enabled
            if self.log_to_wandb and grad_stats['global_stats']:
                try:
                    import wandb
                    
                    log_dict = {
                        f'gradients/global_{k}': v 
                        for k, v in grad_stats['global_stats'].items() 
                        if isinstance(v, (int, float))
                    }
                    
                    # Add percentiles
                    if 'percentiles' in grad_stats['global_stats']:
                        for k, v in grad_stats['global_stats']['percentiles'].items():
                            log_dict[f'gradients/{k}'] = v
                    
                    # Add layer group stats
                    for group, stats in grad_stats.get('layer_group_stats', {}).items():
                        for k, v in stats.items():
                            log_dict[f'gradients/{group}_{k}'] = v
                    
                    wandb.log(log_dict, step=state.global_step)
                    
                except Exception as e:
                    print(f"Warning: Could not log to wandb: {e}")
            
            # Store in memory (limited to prevent memory issues)
            if len(self.gradient_stats) < 1000:
                self.gradient_stats.append(grad_stats)
                
        except Exception as e:
            print(f"Warning: Error collecting gradient stats at step {state.global_step}: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate summary plots and statistics at the end of training."""
        print(f"\nGradient monitoring complete. Logs saved to: {self.grad_log_file}")
        
        # Generate summary visualization
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate a summary plot of gradient statistics over training."""
        try:
            import matplotlib.pyplot as plt
            
            # Load all gradient stats from file
            stats = []
            if self.grad_log_file.exists():
                with open(self.grad_log_file, 'r') as f:
                    for line in f:
                        stats.append(json.loads(line))
            
            if not stats:
                print("No gradient statistics to plot.")
                return
            
            # Extract data for plotting
            steps = [s['step'] for s in stats if s.get('global_stats')]
            norms = [s['global_stats']['norm'] for s in stats if s.get('global_stats')]
            means = [abs(s['global_stats']['mean']) for s in stats if s.get('global_stats')]
            maxs = [s['global_stats']['max'] for s in stats if s.get('global_stats')]
            stds = [s['global_stats']['std'] for s in stats if s.get('global_stats')]
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Gradient Norm
            axes[0, 0].plot(steps, norms, 'b-', linewidth=1.5, alpha=0.8)
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Gradient Norm')
            axes[0, 0].set_title('Global Gradient Norm')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Max Gradient
            axes[0, 1].plot(steps, maxs, 'r-', linewidth=1.5, alpha=0.8)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Max |Gradient|')
            axes[0, 1].set_title('Maximum Absolute Gradient')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Mean Gradient
            axes[1, 0].plot(steps, means, 'g-', linewidth=1.5, alpha=0.8)
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Mean |Gradient|')
            axes[1, 0].set_title('Mean Absolute Gradient')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Gradient Std
            axes[1, 1].plot(steps, stds, 'm-', linewidth=1.5, alpha=0.8)
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Gradient Std')
            axes[1, 1].set_title('Gradient Standard Deviation')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / 'gradient_summary.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Gradient summary plot saved to: {plot_file}")
            
            plt.close()
            
            # Generate layer-wise comparison if available
            self._plot_layer_comparison(stats)
            
        except Exception as e:
            print(f"Warning: Could not generate summary plots: {e}")
    
    def _plot_layer_comparison(self, stats):
        """Plot comparison of gradient norms across different layer groups."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract layer group data
            steps = [s['step'] for s in stats if s.get('layer_group_stats')]
            
            if not steps:
                return
            
            layer_groups = set()
            for s in stats:
                if s.get('layer_group_stats'):
                    layer_groups.update(s['layer_group_stats'].keys())
            
            if not layer_groups:
                return
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for group in layer_groups:
                norms = []
                group_steps = []
                for s in stats:
                    if s.get('layer_group_stats', {}).get(group):
                        norms.append(s['layer_group_stats'][group]['norm'])
                        group_steps.append(s['step'])
                
                if norms:
                    ax.plot(group_steps, norms, linewidth=2, alpha=0.8, label=group)
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norms by Layer Group')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / 'gradient_layer_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Layer comparison plot saved to: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate layer comparison plot: {e}")
