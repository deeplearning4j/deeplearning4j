/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.factory;

/**
 * ND4J backend Environment instance
 *
 * @author Alex Black
 */
public interface Environment {

    /** BLAS major version number (if applicable) */
    int blasMajorVersion();
    /** BLAS minor version number (if applicable) */
    int blasMinorVersion();
    /** BLAS patch version number (if applicable) */
    int blasPatchVersion();

    /** Returns true if ND4J is set to verbose mode */
    boolean isVerbose();
    /** Set verbose mode */
    void setVerbose(boolean reallyVerbose);
    /** Returns true if ND4J is set to debug mode */
    boolean isDebug();
    /** Returns true if ND4J is set to profiling mode */
    boolean isProfiling();
    /** Returns true if ND4J is set to detecting leaks mode */
    boolean isDetectingLeaks();
    /** Returns true if ND4J is set to debug and verbose mode */
    boolean isDebugAndVerbose();

    /** Set debug mode */
    void setDebug( boolean reallyDebug);
    /** Set profiling mode */
    void setProfiling( boolean reallyProfile);
    /** Set leaks detection mode */
    void setLeaksDetector( boolean reallyDetect);
    /** Returns true if helpers (cuDNN, DNNL/MKLDNN etc) are allowed */
    boolean helpersAllowed();
    /** Set whether helpers (cuDNN, DNNL/MKLDNN etc) are allowed */
    void allowHelpers(boolean reallyAllow);

    /** Returns the TAD (tensor along dimension) threshold for ops */
    int tadThreshold();
    /** Set the TAD (tensor along dimension) threshold for ops */
    void setTadThreshold(int threshold);

    /** Returns the elementwise threshold for ops */
    int elementwiseThreshold();
    /** Set the elementwise threshold for ops */
    void setElementwiseThreshold(int threshold);

    /** Returns the maximum number of threads for C++ op execution (if applicable) */
    int maxThreads();
    /** Set the maximum number of threads for C++ op execution (if applicable) */
    void setMaxThreads(int max);

    /** Returns the maximum number of master threads for C++ op execution (if applicable) */
    int maxMasterThreads();
    /** Set the maximum number of master threads for C++ op execution (if applicable) */
    void setMaxMasterThreads(int max);

    /** Set the maximum primary memory */
    void setMaxPrimaryMemory(long maxBytes);
    /** Set the maximum special memory */
    void setMaxSpecialMemory(long maxBytes);
    /** Set the maximum device memory */
    void setMaxDeviceMemory(long maxBytes);

    /** Return true if the backend is a CPU backend, or false otherwise */
    boolean isCPU();

    /**
     * This method allows to set memory limit for a specific group of devices. I.e. CUDA or CPU
     * @param group
     * @param numBytes
     */
    void setGroupLimit(int group, long numBytes);

    /**
     * This method allows to set memory limit for a specific device. I.e. GPU_0
     * @param deviceId
     * @param numBytes
     */
    void setDeviceLimit(int deviceId, long numBytes);

    /**
     * This method returns current group limit
     * @param group
     * @return
     */
    long getGroupLimit(int group);

    /**
     * This method returns current device limit
     * @param deviceId
     * @return
     */
    long getDeviceLimit(int deviceId);

    /**
     * This method returns current allocated amount for a specific device. I.e. GPU_0
     * @param deviceId
     * @return
     */
    long getDeviceCouner(int deviceId);
}
