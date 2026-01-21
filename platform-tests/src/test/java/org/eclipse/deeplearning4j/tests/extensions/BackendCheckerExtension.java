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
package org.eclipse.deeplearning4j.tests.extensions;

import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Set;

/**
 * This extension manages test execution based on backend and helper configuration.
 *
 * Features:
 * - Disables resource-intensive tests on GPU backends
 * - Integrates with MultiBackendTestConfiguration for helper-aware testing
 * - Supports helper-specific test filtering via tags
 * - Provides detailed logging for test decisions
 *
 * System Properties:
 * - sd.test.skip.large: Skip large resource tests (default: true for GPU)
 * - sd.test.skip.downloads: Skip download tests (default: true for GPU)
 * - sd.test.allow.all: Allow all tests regardless of backend (default: false)
 *
 * @author Adam Gibson
 * @author Eclipse Deeplearning4j Development Team
 */
public class BackendCheckerExtension implements ExecutionCondition {

    private static final Logger log = LoggerFactory.getLogger(BackendCheckerExtension.class);

    // Tags for helper-specific tests
    public static final String TAG_ONEDNN = "onednn";
    public static final String TAG_CUDNN = "cudnn";
    public static final String TAG_ARMCOMPUTE = "armcompute";
    public static final String TAG_MPS = "mps";
    public static final String TAG_ACCELERATE = "accelerate";
    public static final String TAG_MLIR = "mlir";
    public static final String TAG_MULTI_BACKEND = "multi-backend";

    // Helper-specific tag set
    public static final Set<String> HELPER_TAGS = Set.of(
            TAG_ONEDNN, TAG_CUDNN, TAG_ARMCOMPUTE, TAG_MPS, TAG_ACCELERATE, TAG_MLIR
    );

    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
        // Check if all tests should be allowed
        if (Boolean.getBoolean("sd.test.allow.all")) {
            return ConditionEvaluationResult.enabled("All tests allowed via sd.test.allow.all");
        }

        Set<String> testTags = context.getTags();
        String testName = context.getDisplayName();

        // Check helper-specific tags
        ConditionEvaluationResult helperResult = evaluateHelperTags(testTags, testName);
        if (helperResult != null && !helperResult.isDisabled()) {
            // Helper tag matched and is enabled
            return helperResult;
        } else if (helperResult != null && helperResult.isDisabled()) {
            // Helper tag matched but helper is not available
            return helperResult;
        }

        // Check for skip flags
        if (shouldSkipLargeResources() && testTags.contains(TagNames.LARGE_RESOURCES)) {
            return ConditionEvaluationResult.disabled("Large resource tests skipped via sd.test.skip.large");
        }

        if (shouldSkipDownloads() && testTags.contains(TagNames.DOWNLOADS)) {
            return ConditionEvaluationResult.disabled("Download tests skipped via sd.test.skip.downloads");
        }

        return ConditionEvaluationResult.enabled("Test enabled by BackendCheckerExtension");
    }

    /**
     * Evaluate helper-specific tags against available helpers
     */
    private ConditionEvaluationResult evaluateHelperTags(Set<String> testTags, String testName) {
        // Check if test has any helper-specific tags
        Set<String> matchingHelperTags = getMatchingTags(testTags, HELPER_TAGS);

        if (matchingHelperTags.isEmpty()) {
            // No helper-specific tags, let other checks handle it
            return null;
        }

        try {
            MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();

            // Check if any of the required helpers are available
            for (String helperTag : matchingHelperTags) {
                MultiBackendTestConfiguration.Helper helper =
                        MultiBackendTestConfiguration.Helper.fromString(helperTag);

                if (helper != null && config.isHelperEnabled(helper)) {
                    log.debug("Test '{}' enabled - helper '{}' is available", testName, helperTag);
                    return ConditionEvaluationResult.enabled(
                            String.format("Helper '%s' is available", helperTag));
                }
            }

            // None of the required helpers are available
            String reason = String.format(
                    "Test '%s' requires one of helpers %s, but none are enabled. Enabled: %s",
                    testName, matchingHelperTags, config.getEnabledHelpers());
            log.debug(reason);
            return ConditionEvaluationResult.disabled(reason);

        } catch (Exception e) {
            log.warn("Error checking helper availability, allowing test: {}", e.getMessage());
            return null;
        }
    }

    private Set<String> getMatchingTags(Set<String> testTags, Set<String> checkTags) {
        Set<String> matching = new HashSet<>();
        for (String tag : testTags) {
            if (checkTags.contains(tag.toLowerCase())) {
                matching.add(tag);
            }
        }
        return matching;
    }

    private boolean shouldSkipLargeResources() {
        String value = System.getProperty("sd.test.skip.large");
        if (value != null) {
            return Boolean.parseBoolean(value);
        }
        // Default: skip on GPU
        return !Nd4j.getEnvironment().isCPU();
    }

    private boolean shouldSkipDownloads() {
        String value = System.getProperty("sd.test.skip.downloads");
        if (value != null) {
            return Boolean.parseBoolean(value);
        }
        // Default: skip on GPU
        return !Nd4j.getEnvironment().isCPU();
    }

    /**
     * Check if a specific helper is available for testing
     */
    public static boolean isHelperAvailable(String helperName) {
        try {
            MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();
            MultiBackendTestConfiguration.Helper helper =
                    MultiBackendTestConfiguration.Helper.fromString(helperName);
            return helper != null && config.isHelperEnabled(helper);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get set of all enabled helpers
     */
    public static Set<String> getEnabledHelpers() {
        try {
            MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();
            Set<String> helpers = new HashSet<>();
            for (MultiBackendTestConfiguration.Helper h : config.getEnabledHelpers()) {
                helpers.add(h.getId());
            }
            return helpers;
        } catch (Exception e) {
            return Set.of("cpu");
        }
    }

    /**
     * Get the current backend name
     */
    public static String getCurrentBackend() {
        try {
            MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();
            return config.getCurrentBackend().getId();
        } catch (Exception e) {
            return Nd4j.getEnvironment().isCPU() ? "cpu" : "gpu";
        }
    }
}
