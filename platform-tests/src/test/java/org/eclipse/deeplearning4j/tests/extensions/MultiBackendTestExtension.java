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

import org.junit.jupiter.api.extension.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Stream;

/**
 * JUnit 5 extension for multi-backend testing.
 *
 * This extension:
 * - Handles @BackendTest annotation processing
 * - Enables/disables tests based on available helpers
 * - Provides test parameterization for multiple helpers
 * - Tracks test results per helper for analysis
 *
 * @author Adam Gibson
 */
public class MultiBackendTestExtension implements
        ExecutionCondition,
        BeforeEachCallback,
        AfterEachCallback,
        TestTemplateInvocationContextProvider {

    private static final Logger log = LoggerFactory.getLogger(MultiBackendTestExtension.class);

    private static final ExtensionContext.Namespace NAMESPACE =
            ExtensionContext.Namespace.create(MultiBackendTestExtension.class);

    // Keys for storing state in extension context
    private static final String KEY_CURRENT_HELPER = "currentHelper";
    private static final String KEY_TEST_START_TIME = "testStartTime";
    private static final String KEY_TEST_RESULTS = "testResults";

    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
        Optional<BackendTest> annotation = findAnnotation(context);

        if (annotation.isEmpty()) {
            return ConditionEvaluationResult.enabled("No @BackendTest annotation present");
        }

        BackendTest backendTest = annotation.get();
        MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();

        // Check backend requirement
        if (backendTest.backends().length > 0) {
            boolean backendMatch = Arrays.stream(backendTest.backends())
                    .anyMatch(b -> b == config.getCurrentBackend());
            if (!backendMatch) {
                return ConditionEvaluationResult.disabled(
                        String.format("Test requires backends %s, but current backend is %s",
                                Arrays.toString(backendTest.backends()),
                                config.getCurrentBackend()));
            }
        }

        // Check helper requirements
        if (backendTest.helpers().length > 0) {
            Set<Helper> requiredHelpers = new HashSet<>();
            for (String h : backendTest.helpers()) {
                Helper helper =
                        Helper.fromString(h);
                if (helper != null) {
                    requiredHelpers.add(helper);
                }
            }

            boolean anyHelperAvailable = requiredHelpers.stream()
                    .anyMatch(config::isHelperEnabled);

            if (!anyHelperAvailable) {
                return ConditionEvaluationResult.disabled(
                        String.format("Test requires helpers %s, but none are available. Enabled: %s",
                                requiredHelpers, config.getEnabledHelpers()));
            }
        }

        // Check excluded helpers
        if (backendTest.excludeHelpers().length > 0) {
            for (String h : backendTest.excludeHelpers()) {
                Helper helper =
                        Helper.fromString(h);
                if (helper != null && config.isHelperEnabled(helper)) {
                    // Check if this is the only enabled helper
                    if (config.getEnabledHelpers().size() == 1) {
                        return ConditionEvaluationResult.disabled(
                                String.format("Test excludes helper %s which is the only enabled helper", h));
                    }
                }
            }
        }

        // Check capability requirement
        String requiredCapability = backendTest.requiresCapability();
        if (!requiredCapability.isEmpty()) {
            Set<Helper> capableHelpers =
                    config.getHelpersWithCapability(requiredCapability);
            if (capableHelpers.isEmpty()) {
                return ConditionEvaluationResult.disabled(
                        String.format("Test requires capability '%s', but no enabled helper supports it",
                                requiredCapability));
            }
        }

        // Check memory requirement
        if (backendTest.minMemoryMB() > 0) {
            long availableMemory = Runtime.getRuntime().maxMemory() / (1024 * 1024);
            if (availableMemory < backendTest.minMemoryMB()) {
                return ConditionEvaluationResult.disabled(
                        String.format("Test requires %d MB memory, but only %d MB available",
                                backendTest.minMemoryMB(), availableMemory));
            }
        }

        return ConditionEvaluationResult.enabled("All backend requirements met");
    }

    @Override
    public void beforeEach(ExtensionContext context) {
        // Record test start time
        context.getStore(NAMESPACE).put(KEY_TEST_START_TIME, System.currentTimeMillis());

        // Log test execution with backend info
        MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();
        String testName = context.getDisplayName();

        log.info("Starting test '{}' on backend {} with helpers {}",
                testName, config.getCurrentBackend(), config.getEnabledHelpers());
    }

    @Override
    public void afterEach(ExtensionContext context) {
        // Calculate test duration
        Long startTime = context.getStore(NAMESPACE).get(KEY_TEST_START_TIME, Long.class);
        long duration = startTime != null ? System.currentTimeMillis() - startTime : 0;

        // Record test result
        MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();
        String testName = context.getDisplayName();

        boolean passed = context.getExecutionException().isEmpty();

        log.info("Completed test '{}' on backend {} - {} ({}ms)",
                testName, config.getCurrentBackend(),
                passed ? "PASSED" : "FAILED", duration);

        // Store result for aggregation
        recordTestResult(context, testName, config.getCurrentBackend().getId(), passed, duration);
    }

    @Override
    public boolean supportsTestTemplate(ExtensionContext context) {
        return findAnnotation(context).isPresent();
    }

    @Override
    public Stream<TestTemplateInvocationContext> provideTestTemplateInvocationContexts(
            ExtensionContext context) {

        Optional<BackendTest> annotation = findAnnotation(context);
        if (annotation.isEmpty()) {
            return Stream.empty();
        }

        BackendTest backendTest = annotation.get();
        MultiBackendTestConfiguration config = MultiBackendTestConfiguration.getInstance();

        // Determine which helpers to test
        Set<Helper> helpersToTest = new HashSet<>();

        if (backendTest.helpers().length > 0) {
            // Specific helpers requested
            for (String h : backendTest.helpers()) {
                Helper helper =
                        Helper.fromString(h);
                if (helper != null && config.isHelperEnabled(helper)) {
                    helpersToTest.add(helper);
                }
            }
        } else {
            // All enabled helpers
            helpersToTest.addAll(config.getEnabledHelpers());
        }

        // Remove excluded helpers
        for (String h : backendTest.excludeHelpers()) {
            Helper helper =
                    Helper.fromString(h);
            if (helper != null) {
                helpersToTest.remove(helper);
            }
        }

        // Filter by capability
        String requiredCapability = backendTest.requiresCapability();
        if (!requiredCapability.isEmpty()) {
            Set<Helper> capableHelpers =
                    config.getHelpersWithCapability(requiredCapability);
            helpersToTest.retainAll(capableHelpers);
        }

        // Create invocation contexts for each helper
        return helpersToTest.stream()
                .map(helper -> new HelperInvocationContext(helper, config.getHelperCapabilities(helper)));
    }

    private Optional<BackendTest> findAnnotation(ExtensionContext context) {
        // Check method first, then class
        Optional<Method> testMethod = context.getTestMethod();
        if (testMethod.isPresent()) {
            BackendTest annotation = testMethod.get().getAnnotation(BackendTest.class);
            if (annotation != null) {
                return Optional.of(annotation);
            }
        }

        Optional<Class<?>> testClass = context.getTestClass();
        if (testClass.isPresent()) {
            BackendTest annotation = testClass.get().getAnnotation(BackendTest.class);
            if (annotation != null) {
                return Optional.of(annotation);
            }
        }

        return Optional.empty();
    }

    private void recordTestResult(ExtensionContext context, String testName,
                                  String backend, boolean passed, long duration) {
        // Get or create results map
        ExtensionContext rootContext = context.getRoot();
        @SuppressWarnings("unchecked")
        Map<String, TestResult> results = rootContext.getStore(NAMESPACE)
                .getOrComputeIfAbsent(KEY_TEST_RESULTS, k -> new HashMap<>(), Map.class);

        String key = testName + "@" + backend;
        results.put(key, new TestResult(testName, backend, passed, duration));
    }

    /**
     * Test result record for aggregation
     */
    public static class TestResult {
        private final String testName;
        private final String backend;
        private final boolean passed;
        private final long durationMs;

        public TestResult(String testName, String backend, boolean passed, long durationMs) {
            this.testName = testName;
            this.backend = backend;
            this.passed = passed;
            this.durationMs = durationMs;
        }

        public String getTestName() { return testName; }
        public String getBackend() { return backend; }
        public boolean isPassed() { return passed; }
        public long getDurationMs() { return durationMs; }
    }

    /**
     * Invocation context for a specific helper
     */
    private static class HelperInvocationContext implements TestTemplateInvocationContext {

        private final Helper helper;
        private final HelperCapabilities capabilities;

        public HelperInvocationContext(Helper helper,
                                       HelperCapabilities capabilities) {
            this.helper = helper;
            this.capabilities = capabilities;
        }

        @Override
        public String getDisplayName(int invocationIndex) {
            return "[" + helper.getId() + "]";
        }

        @Override
        public List<Extension> getAdditionalExtensions() {
            return List.of(
                    new HelperParameterResolver(helper, capabilities),
                    new HelperSetupCallback(helper)
            );
        }
    }

    /**
     * Parameter resolver for injecting helper info into tests
     */
    private static class HelperParameterResolver implements ParameterResolver {

        private final Helper helper;
        private final HelperCapabilities capabilities;

        public HelperParameterResolver(Helper helper,
                                       HelperCapabilities capabilities) {
            this.helper = helper;
            this.capabilities = capabilities;
        }

        @Override
        public boolean supportsParameter(ParameterContext parameterContext,
                                         ExtensionContext extensionContext) {
            Class<?> type = parameterContext.getParameter().getType();
            return type == Helper.class ||
                    type == HelperCapabilities.class ||
                    type == HelperTestConfig.class;
        }

        @Override
        public Object resolveParameter(ParameterContext parameterContext,
                                        ExtensionContext extensionContext) {
            Class<?> type = parameterContext.getParameter().getType();
            if (type == Helper.class) {
                return helper;
            } else if (type == HelperCapabilities.class) {
                return capabilities;
            } else if (type == HelperTestConfig.class) {
                return new HelperTestConfig(helper, capabilities);
            }
            return null;
        }
    }

    /**
     * Callback to set up the helper before each test invocation
     */
    private static class HelperSetupCallback implements BeforeEachCallback {

        private final Helper helper;

        public HelperSetupCallback(Helper helper) {
            this.helper = helper;
        }

        @Override
        public void beforeEach(ExtensionContext context) {
            // Store current helper in context for other extensions to use
            context.getStore(NAMESPACE).put(KEY_CURRENT_HELPER, helper);

            // Set environment variable for native code to pick up
            System.setProperty("sd.preferred.helper", helper.getId());

            log.debug("Set up helper {} for test {}", helper, context.getDisplayName());
        }
    }
}
