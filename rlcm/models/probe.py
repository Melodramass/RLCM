"""
A lightweight 2-Layer MLP probe for confidence extraction.

The probe takes hidden states from a language model and predicts
a confidence logit (use sigmoid to get probability).

Architecture: Linear -> ReLU -> Linear (outputs logits)
"""

import torch
import torch.nn as nn
from typing import Optional


class ConfProbe(nn.Module):
    """
    A simple 2-layer MLP probe for extracting confidence from hidden states.
    
    Args:
        hidden_size: Dimension of the input hidden states (e.g., 1536 for 1.5B, 3584 for 7B)
        intermediate_size: Dimension of the hidden layer (default: hidden_size // 4)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size // 4
        
        self.probe = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the confidence probe.
        
        Args:
            hidden_states: Hidden states from the language model.
                Shape: (batch_size, hidden_size) or (batch_size, seq_len, hidden_size)
                If 3D, uses the last token's hidden state.
        
        Returns:
            Logits (apply sigmoid to get confidence in [0, 1]). Shape: (batch_size,)
        """
        # If 3D tensor, extract last token's hidden state
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]  # (batch_size, hidden_size)
        
        confidence = self.probe(hidden_states).squeeze(-1)  # (batch_size,)
        return confidence
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "ConfProbe":
        """Load a pretrained probe from a checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        
        # Infer hidden_size from the first layer's weight
        hidden_size = state_dict["probe.0.weight"].shape[1]
        intermediate_size = state_dict["probe.0.weight"].shape[0]
        
        model = cls(hidden_size=hidden_size, intermediate_size=intermediate_size, **kwargs)
        model.load_state_dict(state_dict)
        return model
    
    def save_pretrained(self, path: str):
        """Save the probe to a checkpoint."""
        torch.save(self.state_dict(), path)

class HiddenStateHook:
    """Hook to capture hidden states from model layers.

    Works with both final layer norm (output is a tensor) and decoder
    layers (output is a tuple where the first element is hidden states).
    """
    
    def __init__(self):
        self.hidden_states = None
    
    def __call__(self, module, input, output):
        # Decoder layers return a tuple (hidden_states, ...) whereas
        # layer norms return a plain tensor.
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # Shape: (batch, seq_len, hidden_size) or (1, total_nnz, hidden_size)
        # Only detach here; moving to CPU is done after all GPU ops complete
        self.hidden_states = hs.detach()
    
    def clear(self):
        self.hidden_states = None