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

/**
 * JNI bridge for NativeMultiBackendWorkspace
 *
 * This file provides JNI implementations that bridge Java native methods
 * to the C++ MultiBackendWorkspace API.
 */

#include <jni.h>
#include <memory/MultiBackendWorkspace.h>
#include <memory/DeviceWorkspaceManager.h>
#include <string>

using namespace sd::memory;

extern "C" {

/**
 * Helper to convert Java string to std::string
 */
static std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (jstr == nullptr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

/**
 * Create a new multi-backend workspace
 */
JNIEXPORT jlong JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeCreate(
    JNIEnv* env,
    jclass clazz,
    jlong initialSize,
    jlong maxSize,
    jboolean crossDeviceMirroring,
    jboolean asyncTransfers,
    jint primaryDeviceType,
    jint primaryDeviceIndex,
    jstring id) {

    try {
        MultiBackendWorkspaceConfig config;
        config.initialSize = static_cast<sd::LongType>(initialSize);
        config.maxSize = static_cast<sd::LongType>(maxSize);
        config.crossDeviceMirroring = crossDeviceMirroring;
        config.asyncTransfers = asyncTransfers;
        config.primaryDevice = DeviceDescriptor(
            static_cast<DeviceType>(primaryDeviceType),
            static_cast<int>(primaryDeviceIndex)
        );

        std::string idStr = jstringToString(env, id);
        MultiBackendWorkspace* workspace = new MultiBackendWorkspace(config, idStr);

        return reinterpret_cast<jlong>(workspace);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return 0;
    }
}

/**
 * Destroy a multi-backend workspace
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeDestroy(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        workspace->destroy();
        delete workspace;
    } catch (const std::exception& e) {
        // Log but don't throw during destruction
    }
}

/**
 * Allocate bytes on primary device
 */
JNIEXPORT jlong JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeAllocateBytes(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jlong numBytes) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        void* ptr = workspace->allocateBytes(static_cast<sd::LongType>(numBytes));
        return reinterpret_cast<jlong>(ptr);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return 0;
    }
}

/**
 * Allocate bytes on specific device
 */
JNIEXPORT jlong JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeAllocateBytesOnDevice(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jlong numBytes,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        void* ptr = workspace->allocateBytes(device, static_cast<sd::LongType>(numBytes));
        return reinterpret_cast<jlong>(ptr);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return 0;
    }
}

/**
 * Enter workspace scope
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeScopeIn(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        workspace->scopeIn();
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Exit workspace scope
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeScopeOut(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        workspace->scopeOut();
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Check if scope is active
 */
JNIEXPORT jboolean JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeIsScopeActive(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return JNI_FALSE;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        return workspace->isScopeActive() ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        return JNI_FALSE;
    }
}

/**
 * Get coherence state for a device
 */
JNIEXPORT jint JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeGetCoherenceState(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        return static_cast<jint>(workspace->getCoherenceState(device));
    } catch (const std::exception& e) {
        return 0;
    }
}

/**
 * Set coherence state for a device
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeSetCoherenceState(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex,
    jint state) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        workspace->setCoherenceState(device, static_cast<CoherenceState>(state));
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Mark data as modified on a device
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeMarkModified(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        workspace->markModified(device);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Transfer data between devices
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeTransferTo(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint srcDeviceType,
    jint srcDeviceIndex,
    jint dstDeviceType,
    jint dstDeviceIndex) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor src(static_cast<DeviceType>(srcDeviceType), srcDeviceIndex);
        DeviceDescriptor dst(static_cast<DeviceType>(dstDeviceType), dstDeviceIndex);
        workspace->transferTo(src, dst);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Ensure data is valid on a device
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeEnsureValidOn(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        workspace->ensureValidOn(device);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Get total allocated size
 */
JNIEXPORT jlong JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeGetTotalAllocatedSize(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        return static_cast<jlong>(workspace->getTotalAllocatedSize());
    } catch (const std::exception& e) {
        return 0;
    }
}

/**
 * Get allocated size on a device
 */
JNIEXPORT jlong JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeGetAllocatedSizeOnDevice(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        return static_cast<jlong>(workspace->getAllocatedSizeOnDevice(device));
    } catch (const std::exception& e) {
        return 0;
    }
}

/**
 * Get current offset
 */
JNIEXPORT jlong JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeGetCurrentOffset(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        return static_cast<jlong>(workspace->getCurrentOffset());
    } catch (const std::exception& e) {
        return 0;
    }
}

/**
 * Release memory on a device
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeReleaseOnDevice(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        workspace->releaseOnDevice(device);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Synchronize a device
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeSyncDevice(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        workspace->syncDevice(device);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Synchronize all devices
 */
JNIEXPORT void JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeSyncAllDevices(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        workspace->syncAllDevices();
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

/**
 * Get number of active devices
 */
JNIEXPORT jint JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeGetActiveDeviceCount(
    JNIEnv* env,
    jclass clazz,
    jlong handle) {

    if (handle == 0) return 0;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        return static_cast<jint>(workspace->getActiveDevices().size());
    } catch (const std::exception& e) {
        return 0;
    }
}

/**
 * Check if device has allocation
 */
JNIEXPORT jboolean JNICALL
Java_org_nd4j_linalg_api_memory_abstracts_NativeMultiBackendWorkspace_nativeHasDeviceAllocation(
    JNIEnv* env,
    jclass clazz,
    jlong handle,
    jint deviceType,
    jint deviceIndex) {

    if (handle == 0) return JNI_FALSE;

    try {
        MultiBackendWorkspace* workspace = reinterpret_cast<MultiBackendWorkspace*>(handle);
        DeviceDescriptor device(static_cast<DeviceType>(deviceType), deviceIndex);
        return workspace->hasDeviceAllocation(device) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        return JNI_FALSE;
    }
}

}  // extern "C"
