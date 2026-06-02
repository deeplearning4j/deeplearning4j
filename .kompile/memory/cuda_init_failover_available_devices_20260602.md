---
name: cuda_init_failover_available_devices_20260602
description: 
type: project
---

DL4J CUDA init/failover fix for subprocess failure where device 0 OOM during CUDA init and later JCublasNDArrayFactory.createBlas/lastErrorCode failed with setCurrentDevice error. Key fixes: Environment_initCuda and initializeDevicesAndFunctions are metadata-only and do not cudaSetDevice/cudaDeviceSynchronize during early native bootstrap; Configuration.updateDevice publishes auto-detected available devices to native immediately via setAvailableDevices; AffinityManager::setAvailableDevices selects the first usable configured device and updates native thread affinity; LaunchContext::defaultContext lazily initializes only the current device context instead of eagerly touching all physical GPUs; P2P detection now reports real configured topology, while non-P2P cross-device transfers/failover remain enabled via migration/managed memory. Validation: full CUDA native/backend build passed, Java nd4j-cuda rebuild passed, CrossDeviceTransferTest#testBinaryOpCrossDeviceInputs passed after config publish.
