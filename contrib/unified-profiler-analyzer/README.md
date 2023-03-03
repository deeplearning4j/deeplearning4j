# Event Logger Analyzer.

This proof of concept tool is for analyzing event logs from the EventLogger and UnifiedProfiler.
When trying to debug off heap memory leaks it can be hard to understand the allocation patterns.

The UnifiedProfiler emits logs in both an aggregate mode and singular mode which can 
either output just runtime and aggregate metrics for workspaces or individual allocation
events for further analysis.

This folder contains classes for helping analyze those logs by converting them to arrow
for some basic visualization or import in to a proper analytics database for analysis.