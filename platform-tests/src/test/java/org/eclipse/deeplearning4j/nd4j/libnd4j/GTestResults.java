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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Container for all GTest results.
 */
public class GTestResults {
    private String name;
    private int tests;
    private int failures;
    private int disabled;
    private int errors;
    private double time;
    private List<GTestSuite> testSuites = new ArrayList<>();

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public int getTests() { return tests; }
    public void setTests(int tests) { this.tests = tests; }

    public int getFailures() { return failures; }
    public void setFailures(int failures) { this.failures = failures; }

    public int getDisabled() { return disabled; }
    public void setDisabled(int disabled) { this.disabled = disabled; }

    public int getErrors() { return errors; }
    public void setErrors(int errors) { this.errors = errors; }

    public double getTime() { return time; }
    public void setTime(double time) { this.time = time; }

    public List<GTestSuite> getTestSuites() { return Collections.unmodifiableList(testSuites); }
    public void addTestSuite(GTestSuite suite) { testSuites.add(suite); }

    public boolean hasFailures() { return failures > 0 || errors > 0; }

    public int getTotalTestCount() {
        return testSuites.stream().mapToInt(GTestSuite::getTests).sum();
    }

    public int getTotalFailureCount() {
        return testSuites.stream().mapToInt(GTestSuite::getFailures).sum();
    }

    public List<GTestCase> getAllFailedTests() {
        List<GTestCase> failed = new ArrayList<>();
        for (GTestSuite suite : testSuites) {
            for (GTestCase tc : suite.getTestCases()) {
                if (tc.isFailure()) {
                    failed.add(tc);
                }
            }
        }
        return failed;
    }

    @Override
    public String toString() {
        return String.format("GTestResults{name='%s', tests=%d, failures=%d, errors=%d, time=%.3fs}",
                name, tests, failures, errors, time);
    }
}
