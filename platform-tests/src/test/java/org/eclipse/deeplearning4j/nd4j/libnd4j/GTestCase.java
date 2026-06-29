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

package org.eclipse.deeplearning4j.nd4j.libnd4j;

/**
 * Represents a single GTest test case.
 */
public class GTestCase {
    private String name;
    private String className;
    private String status;
    private double time;
    private String file;
    private int line;
    private boolean failure;
    private String failureMessage;
    private String failureType;
    private String failureDetails;
    private boolean skipped;
    private String skipReason;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getClassName() { return className; }
    public void setClassName(String className) { this.className = className; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public double getTime() { return time; }
    public void setTime(double time) { this.time = time; }

    public String getFile() { return file; }
    public void setFile(String file) { this.file = file; }

    public int getLine() { return line; }
    public void setLine(int line) { this.line = line; }

    public boolean isFailure() { return failure; }
    public void setFailure(boolean failure) { this.failure = failure; }

    public String getFailureMessage() { return failureMessage; }
    public void setFailureMessage(String failureMessage) { this.failureMessage = failureMessage; }

    public String getFailureType() { return failureType; }
    public void setFailureType(String failureType) { this.failureType = failureType; }

    public String getFailureDetails() { return failureDetails; }
    public void setFailureDetails(String failureDetails) { this.failureDetails = failureDetails; }

    public boolean isSkipped() { return skipped; }
    public void setSkipped(boolean skipped) { this.skipped = skipped; }

    public String getSkipReason() { return skipReason; }
    public void setSkipReason(String skipReason) { this.skipReason = skipReason; }

    public String getFullName() {
        return className + "." + name;
    }

    public boolean isPassed() {
        return !failure && !skipped && "run".equalsIgnoreCase(status);
    }

    @Override
    public String toString() {
        return String.format("GTestCase{name='%s', status='%s', time=%.3fs, failure=%s}",
                getFullName(), status, time, failure);
    }
}
