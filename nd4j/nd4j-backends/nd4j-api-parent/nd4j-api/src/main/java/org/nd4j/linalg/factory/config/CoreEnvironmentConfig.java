/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.factory.config;

/**
 * Core runtime configuration: verbose/debug mode, threading, memory limits,
 * BLAS settings, helpers, and general diagnostics.
 *
 * <p>New code should access these methods via {@code Nd4j.getEnvironment().core().isVerbose()}
 * (once the accessor is wired). Legacy flat access on Environment still works.</p>
 */
public interface CoreEnvironmentConfig {

    boolean isEnableBlas();
    void setEnableBlas(boolean reallyEnable);

    boolean isLogNativeNDArrayCreation();
    void setLogNativeNDArrayCreation(boolean logNativeNDArrayCreation);

    boolean isCheckOutputChange();
    void setCheckOutputChange(boolean reallyCheck);

    boolean isCheckInputChange();
    void setCheckInputChange(boolean reallyCheck);

    void setLogNDArrayEvents(boolean logNDArrayEvents);
    boolean isLogNDArrayEvents();

    boolean isDeleteShapeInfo();
    void setDeleteShapeInfo(boolean reallyDelete);

    boolean isVerbose();
    void setVerbose(boolean reallyVerbose);
    boolean isDebug();
    boolean isProfiling();
    boolean isDetectingLeaks();
    boolean isDebugAndVerbose();
    void setDebug(boolean reallyDebug);
    void setProfiling(boolean reallyProfile);
    void setLeaksDetector(boolean reallyDetect);
    boolean helpersAllowed();
    void allowHelpers(boolean reallyAllow);

    int tadThreshold();
    void setTadThreshold(int threshold);
    int elementwiseThreshold();
    void setElementwiseThreshold(int threshold);

    int maxThreads();
    void setMaxThreads(int max);
    int maxMasterThreads();
    void setMaxMasterThreads(int max);

    void setMaxPrimaryMemory(long maxBytes);
    void setMaxSpecialMemory(long maxBytes);
    void setMaxDeviceMemory(long maxBytes);

    boolean isCPU();

    void setGroupLimit(int group, long numBytes);
    void setDeviceLimit(int deviceId, long numBytes);
    long getGroupLimit(int group);
    long getDeviceLimit(int deviceId);
    long getDeviceCounter(int deviceId);

    boolean isFuncTracePrintDeallocate();
    boolean isFuncTracePrintAllocate();
    void setFuncTraceForDeallocate(boolean reallyTrace);
    void setFuncTraceForAllocate(boolean reallyTrace);

    boolean isDeletePrimary();
    boolean isDeleteSpecial();
    void setDeletePrimary(boolean reallyDelete);
    void setDeleteSpecial(boolean reallyDelete);

    int blasMajorVersion();
    int blasMinorVersion();
    int blasPatchVersion();
}
