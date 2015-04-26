---
title: 
layout: default
---

# Canova: Vectorization Lib for ML Tools

Key aspects:
- [Canova](https://github.com/deeplearning4j/Canova) uses an input/output format system (think like how Hadoop does with MapReduce).
- its designed to support all major types of input data (text, csv, audio, image, and video) with these specific input formats
- it uses an output format system to specifiy an implementation neutral type of vector format (ARFF, SVMLight, etc)
- its designed to run from the command line (bash has the largest user base, more so than any api)
- in the event that you have a specialized input format (think exotic image format, etc), you can write your own custom input format and let the rest of the codebase handle the transformation pipeline
- DL4J's new CLI system will use input formats that match up to not only your custom ones, but also the standard ones (ARFF, SVMLight, etc)

Summary: Vectorization is now a first-class citizen.

Right now we're finishing up the CLI system and running tests on basic datasets for converting just stock CSV data that you'd export from a DB or get from UCI. You can transform this data with a small conf file based transform language (label, normalize, copy, etc).
