/*
 *  ******************************************************************************
 *  *
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
package org.nd4j.linalg.cpu.nativecpu;

import org.nd4j.linalg.factory.Environment;

public class CpuEnvironment implements Environment {


    private static final CpuEnvironment INSTANCE = new CpuEnvironment();


    public static CpuEnvironment getInstance(){
        return INSTANCE;
    }


    @Override
    public int blasMajorVersion() {
        return 0;
    }

    @Override
    public int blasMinorVersion() {
        return 0;
    }

    @Override
    public int blasPatchVersion() {
        return 0;
    }

    @Override
    public boolean isVerbose() {
        return false;
    }

    @Override
    public void setVerbose(boolean reallyVerbose) {

    }

    @Override
    public boolean isDebug() {
        return false;
    }

    @Override
    public boolean isProfiling() {
        return false;
    }

    @Override
    public boolean isDetectingLeaks() {
        return false;
    }

    @Override
    public boolean isDebugAndVerbose() {
        return false;
    }

    @Override
    public void setDebug(boolean reallyDebug) {

    }

    @Override
    public void setProfiling(boolean reallyProfile) {

    }

    @Override
    public void setLeaksDetector(boolean reallyDetect) {

    }

    @Override
    public boolean helpersAllowed() {
        return false;
    }

    @Override
    public void allowHelpers(boolean reallyAllow) {

    }

    @Override
    public int tadThreshold() {
        return 0;
    }

    @Override
    public void setTadThreshold(int threshold) {

    }

    @Override
    public int elementwiseThreshold() {
        return 0;
    }

    @Override
    public void setElementwiseThreshold(int threshold) {

    }

    @Override
    public int maxThreads() {
        return 0;
    }

    @Override
    public void setMaxThreads(int max) {

    }

    @Override
    public int maxMasterThreads() {
        return 0;
    }

    @Override
    public void setMaxMasterThreads(int max) {

    }

    @Override
    public void setMaxPrimaryMemory(long maxBytes) {

    }

    @Override
    public void setMaxSpecialMemory(long maxBytes) {

    }

    @Override
    public void setMaxDeviceMemory(long maxBytes) {

    }

    @Override
    public boolean isCPU() {
        return false;
    }

    @Override
    public void setGroupLimit(int group, long numBytes) {

    }

    @Override
    public void setDeviceLimit(int deviceId, long numBytes) {

    }

    @Override
    public long getGroupLimit(int group) {
        return 0;
    }

    @Override
    public long getDeviceLimit(int deviceId) {
        return 0;
    }

    @Override
    public long getDeviceCounter(int deviceId) {
        return 0;
    }

    @Override
    public boolean isFuncTracePrintDeallocate() {
        return false;
    }

    @Override
    public boolean isFuncTracePrintAllocate() {
        return false;
    }

    @Override
    public void setFuncTraceForDeallocate(boolean reallyTrace) {

    }

    @Override
    public void setFuncTraceForAllocate(boolean reallyTrace) {

    }

    @Override
    public boolean isDeletePrimary() {
        return false;
    }

    @Override
    public boolean isDeleteSpecial() {
        return false;
    }

    @Override
    public void setDeletePrimary(boolean reallyDelete) {

    }

    @Override
    public void setDeleteSpecial(boolean reallyDelete) {

    }
}
