/*
 * *****************************************************************************
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * ****************************************************************************
 */

package org.nd4j.common.tests.results;

import java.util.*;

/**
 * Records named milestones with metrics to persistent storage.
 * <p>
 * Use this to snapshot "at this commit, these results hold" so they don't get lost.
 * Callable from test code, benchmark runners, or standalone.
 *
 * <pre>{@code
 * // From a test or benchmark:
 * TestMilestone.record("DSP core batch: 1590 tests, 0 failures",
 *     "tests", 1590, "failures", 0, "errors", 0, "skipped", 0);
 *
 * // With tok/s metrics:
 * TestMilestone.record("VLM decode baseline",
 *     "lateSteadyTokPerSec", 69.57, "config", "OPTIMAL", "tokens", 250);
 *
 * // Read back all milestones:
 * TestMilestone.list().forEach(System.out::println);
 * }</pre>
 */
public class TestMilestone {

    private TestMilestone() {}

    /**
     * Records a milestone with key-value metric pairs.
     * Keys must be Strings; values can be String, Number, or Boolean.
     *
     * @param description human-readable description of what this milestone records
     * @param kvPairs     alternating key, value pairs: ("tests", 1590, "failures", 0)
     */
    public static void record(String description, Object... kvPairs) {
        Map<String, Object> metrics = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kvPairs.length; i += 2) {
            metrics.put(String.valueOf(kvPairs[i]), kvPairs[i + 1]);
        }
        TestResultStore.appendMilestone(description, metrics);
    }

    /**
     * Records a milestone with a pre-built metrics map.
     */
    public static void record(String description, Map<String, Object> metrics) {
        TestResultStore.appendMilestone(description, metrics);
    }

    /**
     * Returns all recorded milestones as raw JSON lines.
     */
    public static List<String> list() {
        return TestResultStore.readMilestones();
    }

    /**
     * Prints all milestones to stdout in a readable format.
     */
    public static void printAll() {
        List<String> lines = list();
        if (lines.isEmpty()) {
            System.out.println("No milestones recorded at " + TestResultStore.getMilestoneFile());
            return;
        }
        System.out.println("=== Milestones (" + lines.size() + ") from " + TestResultStore.getMilestoneFile() + " ===");
        for (String line : lines) {
            System.out.println(line);
        }
    }
}
