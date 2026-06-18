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

import org.junit.jupiter.api.extension.BeforeAllCallback;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Auto-registered extension that enables debug and/or verbose mode based on
 * system properties, so you never need to embed {@code setDebug(true)} in test code.
 * <p>
 * <b>Usage from Maven:</b>
 * <pre>
 * mvn test -Dtest=SomeTest -Dnd4j.debug=true -Dnd4j.verbose=true
 * </pre>
 * <p>
 * This extension fires once per test class ({@code BeforeAllCallback}).
 * It reads two system properties:
 * <ul>
 *   <li>{@code nd4j.debug} — calls {@code Nd4j.getEnvironment().setDebug(true)}</li>
 *   <li>{@code nd4j.verbose} — calls {@code Nd4j.getEnvironment().setVerbose(true)}</li>
 * </ul>
 * <p>
 * The env var path ({@code ND4J_DEBUG}/​{@code ND4J_VERBOSE}) is already wired
 * through surefire's {@code <environmentVariables>} in the pom. This extension
 * provides a second path via system properties which is more reliable with forked
 * surefire JVMs and works even if the env var wiring didn't fire.
 * <p>
 * Registered via {@code META-INF/services/org.junit.jupiter.api.extension.Extension}.
 */
public class DebugVerboseExtension implements BeforeAllCallback {

    private static volatile boolean initialized = false;

    @Override
    public void beforeAll(ExtensionContext context) {
        if (initialized) return;
        initialized = true;

        try {
            boolean debug = Boolean.parseBoolean(System.getProperty("nd4j.debug", "false"));
            boolean verbose = Boolean.parseBoolean(System.getProperty("nd4j.verbose", "false"));

            if (debug) {
                Nd4j.getEnvironment().setDebug(true);
                System.out.println("[DebugVerboseExtension] Debug mode ENABLED via -Dnd4j.debug=true");
            }
            if (verbose) {
                Nd4j.getEnvironment().setVerbose(true);
                System.out.println("[DebugVerboseExtension] Verbose mode ENABLED via -Dnd4j.verbose=true");
            }
        } catch (Exception e) {
            // Nd4j may not be initialized yet in some edge cases — skip silently
        }
    }
}
