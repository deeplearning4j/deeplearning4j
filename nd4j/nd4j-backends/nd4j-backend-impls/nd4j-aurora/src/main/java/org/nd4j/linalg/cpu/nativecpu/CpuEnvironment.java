/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.factory.Environment;

/**
 * CPU backend implementation of {@link Environment}
 *
 * @author Alex Black
 */
public class CpuEnvironment implements Environment {


    private static final CpuEnvironment INSTANCE = new CpuEnvironment();

    public static CpuEnvironment getInstance(){
        return INSTANCE;
    }

    protected CpuEnvironment(){
    }

    @Override
    public int blasMajorVersion() {
        return 0; // e.blasMajorVersion();
    }

    @Override
    public int blasMinorVersion() {
        return 0; // e.blasMinorVersion();
    }

    @Override
    public int blasPatchVersion() {
        return 0; // e.blasMajorVersion();
    }

    @Override
    public boolean isVerbose() {
        return false; // e.isVerbose();
    }

    @Override
    public void setVerbose(boolean reallyVerbose) {
        // e.setVerbose(reallyVerbose);
    }

    @Override
    public boolean isDebug() {
        return false; // e.isDebug();
    }

    @Override
    public boolean isProfiling() {
        return false; // e.isProfiling();
    }

    @Override
    public boolean isDetectingLeaks() {
        return false; // e.isDetectingLeaks();
    }

    @Override
    public boolean isDebugAndVerbose() {
        return false; // e.isDebugAndVerbose();
    }

    @Override
    public void setDebug(boolean reallyDebug) {
        // e.setDebug(reallyDebug);
    }

    @Override
    public void setProfiling(boolean reallyProfile) {
        // e.setProfiling(reallyProfile);
    }

    @Override
    public void setLeaksDetector(boolean reallyDetect) {
        // e.setLeaksDetector(reallyDetect);
    }

    @Override
    public boolean helpersAllowed() {
        return false; // e.helpersAllowed();
    }

    @Override
    public void allowHelpers(boolean reallyAllow) {
        // e.allowHelpers(reallyAllow);
    }

    @Override
    public int tadThreshold() {
        return 0; // e.tadThreshold();
    }

    @Override
    public void setTadThreshold(int threshold) {
        // e.setTadThreshold(threshold);
    }

    @Override
    public int elementwiseThreshold() {
        return 0; // e.elementwiseThreshold();
    }

    @Override
    public void setElementwiseThreshold(int threshold) {
        // e.setElementwiseThreshold(threshold);
    }

    @Override
    public int maxThreads() {
        return 8; // e.maxThreads();
    }

    @Override
    public void setMaxThreads(int max) {
        // e.setMaxThreads(max);
    }

    @Override
    public int maxMasterThreads() {
        return 8; // e.maxMasterThreads();
    }

    @Override
    public void setMaxMasterThreads(int max) {
        // e.setMaxMasterThreads(max);
    }

    @Override
    public void setMaxPrimaryMemory(long maxBytes) {
        // e.setMaxPrimaryMemory(maxBytes);
    }

    @Override
    public void setMaxSpecialMemory(long maxBytes) {
        // e.setMaxSpecialyMemory(maxBytes);
    }

    @Override
    public void setMaxDeviceMemory(long maxBytes) {
        // e.setMaxDeviceMemory(maxBytes);
    }

    @Override
    public boolean isCPU() {
        return false; // e.isCPU();
    }

    @Override
    public void setGroupLimit(int group, long numBytes) {
        // e.setGroupLimit(group, numBytes);
    }

    @Override
    public void setDeviceLimit(int deviceId, long numBytes) {
        // e.setDeviceLimit(deviceId, numBytes);
    }

    @Override
    public long getGroupLimit(int group) {
        return 0; // e.getGroupLimit(group);
    }

    @Override
    public long getDeviceLimit(int deviceId) {
        return 0; // e.getDeviceLimit(deviceId);
    }

    @Override
    public long getDeviceCouner(int deviceId) {
        return 0; // e.getDeviceCounter(deviceId);
    }
}
