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

package org.eclipse.deeplearning4j.tests.extensions;

import org.junit.jupiter.api.extension.*;
import org.nd4j.common.tests.results.TestResultStore;

import java.lang.reflect.Method;
import java.util.*;

/**
 * Auto-registered JUnit 5 extension that passively records every test result
 * to persistent storage at {@code ~/.dl4j/test-results/}.
 * <p>
 * This extension is registered via
 * {@code META-INF/services/org.junit.jupiter.api.extension.Extension}
 * and requires no annotations or base class changes on test classes.
 * <p>
 * For each test method it records:
 * <ul>
 *   <li>Pass/fail/skip/abort status with duration</li>
 *   <li>JUnit tags</li>
 *   <li>Error message on failure</li>
 *   <li>DSP metrics when DSP diagnostics are active</li>
 * </ul>
 * <p>
 * Disable with {@code -Ddl4j.test.results.disabled=true}.
 */
public class TestResultRecorderExtension implements
        BeforeTestExecutionCallback,
        TestWatcher {

    private static final ExtensionContext.Namespace NS =
            ExtensionContext.Namespace.create(TestResultRecorderExtension.class);
    private static final String START_KEY = "startNanos";

    // ─── Timing capture ─────────────────────────────────────────

    @Override
    public void beforeTestExecution(ExtensionContext context) {
        context.getStore(NS).put(START_KEY, System.nanoTime());
    }

    // ─── Result recording ───────────────────────────────────────

    @Override
    public void testSuccessful(ExtensionContext context) {
        recordResult(context, "PASS", null);
    }

    @Override
    public void testFailed(ExtensionContext context, Throwable cause) {
        recordResult(context, "FAIL", cause);
    }

    @Override
    public void testAborted(ExtensionContext context, Throwable cause) {
        recordResult(context, "ABORT", cause);
    }

    @Override
    public void testDisabled(ExtensionContext context, Optional<String> reason) {
        // For disabled tests, beforeTestExecution never fires — duration is 0
        String testClass = context.getTestClass().map(Class::getName).orElse("unknown");
        String testMethod = context.getTestMethod().map(Method::getName).orElse(null);
        String displayName = context.getDisplayName();
        Set<String> tags = context.getTags();
        String errorMsg = reason.orElse(null);

        TestResultStore.appendResult(testClass, testMethod, displayName,
                "SKIP", 0, tags, errorMsg, null);
    }

    // ─── Internal ───────────────────────────────────────────────

    private void recordResult(ExtensionContext context, String status, Throwable error) {
        try {
            long startNs = context.getStore(NS)
                    .getOrDefault(START_KEY, Long.class, System.nanoTime());
            long durationMs = (System.nanoTime() - startNs) / 1_000_000;

            String testClass = context.getRequiredTestClass().getName();
            String testMethod = context.getRequiredTestMethod().getName();
            String displayName = context.getDisplayName();
            Set<String> tags = context.getTags();
            String errorMsg = error != null ? error.getMessage() : null;

            // Capture DSP metrics if diagnostics are active
            Map<String, Object> dspMetrics = captureDspMetrics();

            TestResultStore.appendResult(testClass, testMethod, displayName,
                    status, durationMs, tags, errorMsg, dspMetrics);
        } catch (Exception e) {
            // Never let recording break tests
        }
    }

    /**
     * Attempts to capture DSP diagnostic metrics via reflection.
     * Returns null if DSP is not active or not available.
     */
    private Map<String, Object> captureDspMetrics() {
        try {
            // Check if DSP diagnostics class is available and has data
            Class<?> diagClass = Class.forName(
                    "org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics");

            // Get step count — if 0, DSP wasn't active for this test
            Object stepCount = diagClass.getMethod("dspDiagGetStepCount").invoke(null);
            if (stepCount == null || ((Number) stepCount).intValue() == 0) {
                return null;
            }

            Map<String, Object> metrics = new LinkedHashMap<>();
            metrics.put("steps", ((Number) stepCount).intValue());

            // Get total event count
            Object eventCount = diagClass.getMethod("dspDiagGetTotalEventCount").invoke(null);
            if (eventCount != null) {
                metrics.put("events", ((Number) eventCount).longValue());
            }

            // Get the plan report text (compact — first 200 chars)
            Object planReport = diagClass.getMethod("dspDiagGetPlanReport").invoke(null);
            if (planReport != null) {
                String report = planReport.toString().trim();
                if (!report.isEmpty()) {
                    // Extract key numbers from the report rather than storing the whole thing
                    metrics.put("hasReport", true);
                }
            }

            return metrics.isEmpty() ? null : metrics;
        } catch (ClassNotFoundException e) {
            // DSP diagnostics not on classpath — normal for some configurations
            return null;
        } catch (Exception e) {
            // Any other reflection error — don't break the test
            return null;
        }
    }
}
