## Centos build for deeplearning4j
Implements a centos 7 based build based on the following docs:
https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idcontainervolumes
https://docs.github.com/en/actions/creating-actions/creating-a-docker-container-action
https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions

This is for a -compat build for older glibcs mainly running on centos based systems.
The action relies on volumes to just run an install workload followed by a deployment
using the host's gpg signature infrastructure very similar to the existing linux-x86_64 builds
based on ubuntu 16.04