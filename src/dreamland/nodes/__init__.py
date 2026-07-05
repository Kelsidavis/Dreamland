"""Dreamland nodes — device capability providers.

A node represents a physical or virtual machine in the Dreamland LAN cluster.
It tracks hardware resources (VRAM, RAM, CPU), loaded models, and active
context windows so the controller can make informed scheduling decisions.
"""

from dreamland.nodes.capability import NodeCapability, NodeResources
from dreamland.nodes.tracker import NodeTracker

__all__ = ["NodeCapability", "NodeResources", "NodeTracker"]
