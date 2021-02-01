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
package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.factory.Environment;
import org.nd4j.nativeblas.Nd4jCpu;

/**
 * CPU backend implementation of {@link Environment}
 *
 * @author Alex Black
 */
public class CpuEnvironment implements Environment {


    private static final CpuEnvironment INSTANCE = new CpuEnvironment(Nd4jCpu.Environment.getInstance());

    private final Nd4jCpu.Environment e;

    public static CpuEnvironment getInstance(){
        return INSTANCE;
    }

    protected CpuEnvironment(Nd4jCpu.Environment environment){
        this.e = environment;
    }

    @Override
    public int blasMajorVersion() {
        return e.blasMajorVersion();
    }

    @Override
    public int blasMinorVersion() {
        return e.blasMinorVersion();
    }

    @Override
    public int blasPatchVersion() {
        return e.blasMajorVersion();
    }

    @Override
    public boolean isVerbose() {
        return e.isVerbose();
    }

    @Override
    public void setVerbose(boolean reallyVerbose) {
        e.setVerbose(reallyVerbose);
    }

    @Override
    public boolean isDebug() {
        return e.isDebug();
    }

    @Override
    public boolean isProfiling() {
        return e.isProfiling();
    }

    @Override
    public boolean isDetectingLeaks() {
        return e.isDetectingLeaks();
    }

    @Override
    public boolean isDebugAndVerbose() {
        return e.isDebugAndVerbose();
    }

    @Override
    public void setDebug(boolean reallyDebug) {
        e.setDebug(reallyDebug);
    }

    @Override
    public void setProfiling(boolean reallyProfile) {
        e.setProfiling(reallyProfile);
    }

    @Override
    public void setLeaksDetector(boolean reallyDetect) {
        e.setLeaksDetector(reallyDetect);
    }

    @Override
    public boolean helpersAllowed() {
        return e.helpersAllowed();
    }

    @Override
    public void allowHelpers(boolean reallyAllow) {
        e.allowHelpers(reallyAllow);
    }

    @Override
    public int tadThreshold() {
        return e.tadThreshold();
    }

    @Override
    public void setTadThreshold(int threshold) {
        e.setTadThreshold(threshold);
    }

    @Override
    public int elementwiseThreshold() {
        return e.elementwiseThreshold();
    }

    @Override
    public void setElementwiseThreshold(int threshold) {
        e.setElementwiseThreshold(threshold);
    }

    @Override
    public int maxThreads() {
        return e.maxThreads();
    }

    @Override
    public void setMaxThreads(int max) {
        e.setMaxThreads(max);
    }

    @Override
    public int maxMasterThreads() {
        return e.maxMasterThreads();
    }

    @Override
    public void setMaxMasterThreads(int max) {
        e.setMaxMasterThreads(max);
    }

    @Override
    public void setMaxPrimaryMemory(long maxBytes) {
        e.setMaxPrimaryMemory(maxBytes);
    }

    @Override
    public void setMaxSpecialMemory(long maxBytes) {
        e.setMaxSpecialyMemory(maxBytes);
    }

    @Override
    public void setMaxDeviceMemory(long maxBytes) {
        e.setMaxDeviceMemory(maxBytes);
    }

    @Override
    public boolean isCPU() {
        return e.isCPU();
    }

    @Override
    public void setGroupLimit(int group, long numBytes) {
        e.setGroupLimit(group, numBytes);
    }

    @Override
    public void setDeviceLimit(int deviceId, long numBytes) {
        e.setDeviceLimit(deviceId, numBytes);
    }

    @Override
    public long getGroupLimit(int group) {
        return e.getGroupLimit(group);
    }

    @Override
    public long getDeviceLimit(int deviceId) {
        return e.getDeviceLimit(deviceId);
    }

    @Override
    public long getDeviceCouner(int deviceId) {
        return e.getDeviceCounter(deviceId);
    }
}
