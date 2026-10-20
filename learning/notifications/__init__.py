"""Optional workflow-completion notifications."""

from learning.notifications.dispatch import send_workflow_notification
from learning.notifications.model import WorkflowNotification

__all__ = ["WorkflowNotification", "send_workflow_notification"]
