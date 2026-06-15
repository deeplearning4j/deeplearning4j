/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//
// Device locality management for LlamaCpp operations
// Ensures operations execute where data resides and manages data movement
// Integrates with ND4J's primary/special buffer system
//

#include "llamacppUtils.h"
#include <array/NDArray.h>
#include <array/DataBuffer.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <system/Environment.h>
#include <mutex>
#include <atomic>

#if HAVE_LLAMACPP

namespace sd {
namespace llamacppUtils {

//////////////////////////////////////////////////////////////////////////
// Data Location Tracking
//////////////////////////////////////////////////////////////////////////

/**
 * Represents where data is currently valid/authoritative
 */
enum class DataLocation {
    UNKNOWN = 0,
    HOST_ONLY = 1,      // Data only valid in primary buffer (CPU)
    DEVICE_ONLY = 2,    // Data only valid in special buffer (GPU)
    SYNCHRONIZED = 3,   // Data valid in both buffers
    STALE = 4           // Data needs refresh from source
};

/**
 * Get the current valid location of an NDArray's data
 */
DataLocation getDataLocation(const NDArray* array) {
    if (array == nullptr || array->isEmpty()) {
        return DataLocation::UNKNOWN;
    }

    auto buffer = array->getDataBuffer();
    if (buffer == nullptr) {
        return DataLocation::UNKNOWN;
    }

    // Check buffer state
    bool primaryValid = buffer->isPrimaryActual();
    bool specialValid = buffer->isSpecialActual();

    if (primaryValid && specialValid) {
        return DataLocation::SYNCHRONIZED;
    } else if (primaryValid) {
        return DataLocation::HOST_ONLY;
    } else if (specialValid) {
        return DataLocation::DEVICE_ONLY;
    }

    return DataLocation::STALE;
}

/**
 * Get string representation of data location
 */
const char* getDataLocationString(DataLocation loc) {
    switch (loc) {
        case DataLocation::HOST_ONLY: return "HOST_ONLY";
        case DataLocation::DEVICE_ONLY: return "DEVICE_ONLY";
        case DataLocation::SYNCHRONIZED: return "SYNCHRONIZED";
        case DataLocation::STALE: return "STALE";
        default: return "UNKNOWN";
    }
}

//////////////////////////////////////////////////////////////////////////
// Locality-Aware Execution Context
//////////////////////////////////////////////////////////////////////////

/**
 * Execution context that tracks data locality for GGML operations
 */
struct LocalityContext {
    int targetDeviceId;
    GgmlBackend targetBackend;
    bool requiresHostData;
    bool requiresDeviceData;
    bool allowDataMovement;
    size_t estimatedMemoryUsage;

    // Statistics
    size_t hostToDeviceTransfers;
    size_t deviceToHostTransfers;
    size_t deviceToDeviceTransfers;
};

static thread_local LocalityContext t_localityContext = {
    -1,                    // targetDeviceId (CPU by default)
    GGML_BACKEND_CPU,     // targetBackend
    true,                  // requiresHostData
    false,                 // requiresDeviceData
    true,                  // allowDataMovement
    0,                     // estimatedMemoryUsage
    0, 0, 0               // transfer counts
};

/**
 * Get the current thread's locality context
 */
LocalityContext& getLocalityContext() {
    return t_localityContext;
}

/**
 * Set up locality context for an operation
 */
void setupLocalityContext(int deviceId, GgmlBackend backend, size_t memoryNeeded) {
    t_localityContext.targetDeviceId = deviceId;
    t_localityContext.targetBackend = backend;
    t_localityContext.requiresHostData = (backend == GGML_BACKEND_CPU);
    t_localityContext.requiresDeviceData = (backend != GGML_BACKEND_CPU);
    t_localityContext.estimatedMemoryUsage = memoryNeeded;
}

//////////////////////////////////////////////////////////////////////////
// Data Synchronization for GGML Operations
//////////////////////////////////////////////////////////////////////////

/**
 * Ensure NDArray data is available on host (primary buffer)
 * Returns pointer to host data or nullptr on failure
 */
void* ensureDataOnHost(NDArray* array, bool forWrite) {
    if (array == nullptr || array->isEmpty()) {
        return nullptr;
    }

    auto buffer = array->getDataBuffer();
    if (buffer == nullptr) {
        return nullptr;
    }

    // Sync to primary (host) buffer
    LaunchContext* ctx = LaunchContext::defaultContext();
    buffer->syncToPrimary(ctx, forWrite);

    if (forWrite) {
        t_localityContext.deviceToHostTransfers++;
    }

    return buffer->primary();
}

/**
 * Ensure NDArray data is available on device (special buffer)
 * Returns pointer to device data or nullptr on failure
 */
void* ensureDataOnDevice(NDArray* array, int deviceId, bool forWrite) {
    if (array == nullptr || array->isEmpty()) {
        return nullptr;
    }

    auto buffer = array->getDataBuffer();
    if (buffer == nullptr) {
        return nullptr;
    }

    // Sync to special (device) buffer
    buffer->syncToSpecial(forWrite);

    if (forWrite) {
        t_localityContext.hostToDeviceTransfers++;
    }

    return buffer->special();
}

/**
 * Prepare an NDArray for a GGML operation based on target backend
 * Handles data synchronization to ensure data is in the right place
 */
bool prepareArrayForGgml(NDArray* array, GgmlBackend backend, bool isInput) {
    if (array == nullptr || array->isEmpty()) {
        return true;  // Nothing to prepare
    }

    DataLocation currentLoc = getDataLocation(array);

    switch (backend) {
        case GGML_BACKEND_CPU:
            // CPU needs data in primary buffer
            if (currentLoc == DataLocation::DEVICE_ONLY) {
                ensureDataOnHost(array, !isInput);
            }
            break;

#if defined(GGML_USE_CUDA)
        case GGML_BACKEND_CUDA:
            // CUDA needs data in special buffer
            if (currentLoc == DataLocation::HOST_ONLY) {
                ensureDataOnDevice(array, array->getDeviceId(), !isInput);
            }
            break;
#endif

        default:
            // For other backends, ensure host data is available
            if (currentLoc == DataLocation::DEVICE_ONLY) {
                ensureDataOnHost(array, !isInput);
            }
            break;
    }

    return true;
}

/**
 * Finalize array after GGML operation (mark output location)
 */
void finalizeArrayAfterGgml(NDArray* array, GgmlBackend backend) {
    if (array == nullptr || array->isEmpty()) {
        return;
    }

    auto buffer = array->getDataBuffer();
    if (buffer == nullptr) {
        return;
    }

    // Mark the appropriate buffer as containing valid data
    switch (backend) {
        case GGML_BACKEND_CPU:
            buffer->writePrimary();
            break;

#if defined(GGML_USE_CUDA)
        case GGML_BACKEND_CUDA:
            buffer->writeSpecial();
            break;
#endif

        default:
            buffer->writePrimary();
            break;
    }
}

//////////////////////////////////////////////////////////////////////////
// Cross-Device Data Movement
//////////////////////////////////////////////////////////////////////////

/**
 * Transfer data between devices
 * Handles the primary/special buffer synchronization
 */
bool transferDataBetweenDevices(NDArray* array, int sourceDevice, int targetDevice) {
    if (array == nullptr || array->isEmpty()) {
        return false;
    }

    if (sourceDevice == targetDevice) {
        return true;  // No transfer needed
    }

    auto buffer = array->getDataBuffer();
    if (buffer == nullptr) {
        return false;
    }

    // For cross-device transfer, we go through host memory
    // Source device -> Host -> Target device

    // Step 1: Ensure data is on host
    LaunchContext* ctx = LaunchContext::defaultContext();
    buffer->syncToPrimary(ctx, false);

    // Step 2: Sync to target device
    // This requires setting up the correct device context
    AffinityManager::setCurrentNativeDevice(targetDevice);
    buffer->syncToSpecial(true);

    t_localityContext.deviceToDeviceTransfers++;

    return true;
}

/**
 * Check if P2P transfer is possible between devices
 */
bool canDoP2PTransfer(int sourceDevice, int targetDevice) {
#if defined(GGML_USE_CUDA)
    // Would check CUDA P2P capability here
    // For now, assume P2P is possible between adjacent devices
    return std::abs(sourceDevice - targetDevice) == 1;
#else
    return false;
#endif
}

//////////////////////////////////////////////////////////////////////////
// GGML Tensor Creation with Locality Awareness
//////////////////////////////////////////////////////////////////////////

/**
 * Create a GGML tensor that wraps NDArray data with proper locality
 * This is the key integration point with ND4J's buffer system
 */
struct ggml_tensor* createGgmlTensorWithLocality(
    struct ggml_context* ctx,
    NDArray* array,
    GgmlBackend backend,
    const char* name
) {
    if (array == nullptr || array->isEmpty()) {
        return nullptr;
    }

    // Prepare data for the target backend
    prepareArrayForGgml(array, backend, true);

    // Get the appropriate data pointer
    void* dataPtr = nullptr;
    switch (backend) {
        case GGML_BACKEND_CPU:
            dataPtr = array->getDataBuffer()->primary();
            break;

#if defined(GGML_USE_CUDA)
        case GGML_BACKEND_CUDA:
            dataPtr = array->getDataBuffer()->special();
            break;
#endif

        default:
            dataPtr = array->getDataBuffer()->primary();
            break;
    }

    if (dataPtr == nullptr) {
        return nullptr;
    }

    // Create GGML tensor
    ggml_type type = toGgmlType(array->dataType());
    int rank = array->rankOf();

    // GGML uses column-major (reversed) dimension order
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < rank && i < 4; i++) {
        ne[i] = array->sizeAt(rank - 1 - i);
    }

    struct ggml_tensor* tensor = ggml_new_tensor(ctx, type, rank, ne);
    if (tensor == nullptr) {
        return nullptr;
    }

    tensor->data = dataPtr;

    if (name != nullptr) {
        ggml_set_name(tensor, name);
    }

    return tensor;
}

/**
 * Copy GGML tensor result back to NDArray with proper locality handling
 */
void copyGgmlResultWithLocality(
    const struct ggml_tensor* tensor,
    NDArray* array,
    GgmlBackend backend
) {
    if (tensor == nullptr || array == nullptr || array->isEmpty()) {
        return;
    }

    auto buffer = array->getDataBuffer();
    if (buffer == nullptr) {
        return;
    }

    // Get destination pointer based on backend
    void* destPtr = nullptr;
    switch (backend) {
        case GGML_BACKEND_CPU:
            destPtr = buffer->primary();
            break;

#if defined(GGML_USE_CUDA)
        case GGML_BACKEND_CUDA:
            destPtr = buffer->special();
            break;
#endif

        default:
            destPtr = buffer->primary();
            break;
    }

    if (destPtr == nullptr || tensor->data == nullptr) {
        return;
    }

    // If tensor data is already in the right place, just update validity
    if (tensor->data == destPtr) {
        finalizeArrayAfterGgml(array, backend);
        return;
    }

    // Copy data and update validity
    size_t dataSize = ggml_nbytes(tensor);
    std::memcpy(destPtr, tensor->data, dataSize);
    finalizeArrayAfterGgml(array, backend);
}

//////////////////////////////////////////////////////////////////////////
// Locality Statistics and Debugging
//////////////////////////////////////////////////////////////////////////

/**
 * Get locality transfer statistics for current thread
 */
void getLocalityStatistics(size_t* h2d, size_t* d2h, size_t* d2d) {
    if (h2d) *h2d = t_localityContext.hostToDeviceTransfers;
    if (d2h) *d2h = t_localityContext.deviceToHostTransfers;
    if (d2d) *d2d = t_localityContext.deviceToDeviceTransfers;
}

/**
 * Reset locality statistics
 */
void resetLocalityStatistics() {
    t_localityContext.hostToDeviceTransfers = 0;
    t_localityContext.deviceToHostTransfers = 0;
    t_localityContext.deviceToDeviceTransfers = 0;
}

/**
 * Print locality information for an array
 */
void printArrayLocalityInfo(const NDArray* array, const char* name) {
    if (!sd::Environment::getInstance().isVerbose()) {
        return;
    }

    if (array == nullptr) {
        printf("%s: NULL\n", name ? name : "array");
        return;
    }

    DataLocation loc = getDataLocation(array);
    int deviceId = array->getDeviceId();

    printf("%s:\n", name ? name : "array");
    printf("  Shape: ");
    for (int i = 0; i < array->rankOf(); i++) {
        printf("%lld ", array->sizeAt(i));
    }
    printf("\n");
    printf("  Device ID: %d\n", deviceId);
    printf("  Data Location: %s\n", getDataLocationString(loc));
    printf("  Data Type: %d\n", static_cast<int>(array->dataType()));

    auto buffer = array->getDataBuffer();
    if (buffer) {
        printf("  Primary Buffer: %p\n", buffer->primary());
        printf("  Special Buffer: %p\n", buffer->special());
        printf("  Primary Actual: %s\n", buffer->isPrimaryActual() ? "yes" : "no");
        printf("  Special Actual: %s\n", buffer->isSpecialActual() ? "yes" : "no");
    }
}

/**
 * Validate that all inputs are on the expected device for an operation
 */
bool validateInputLocality(const std::vector<const NDArray*>& inputs, int expectedDevice) {
    for (const auto* input : inputs) {
        if (input == nullptr || input->isEmpty()) {
            continue;
        }

        int inputDevice = input->getDeviceId();
        if (inputDevice != expectedDevice) {
            if (sd::Environment::getInstance().isVerbose()) {
                printf("WARNING: Input on device %d but operation targets device %d\n",
                       inputDevice, expectedDevice);
            }
            return false;
        }
    }
    return true;
}

}  // namespace llamacppUtils
}  // namespace sd

#endif  // HAVE_LLAMACPP
