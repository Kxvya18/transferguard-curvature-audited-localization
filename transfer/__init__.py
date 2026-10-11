"""Transfer and few-shot adaptation helpers."""

from .adaptation import coral_loss, fit_prototypes, prototypical_logits

__all__ = ["coral_loss", "fit_prototypes", "prototypical_logits"]

