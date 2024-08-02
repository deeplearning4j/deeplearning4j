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
package org.deeplearning4j;

import org.deeplearning4j.nn.conf.ConfClassLoading;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.common.tools.ClassInitializerUtil;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

@DisplayName("Base DL 4 J Test")
public abstract class BaseDL4JTest {


    protected long startTime;

    protected int threadCountBefore;

    private final int DEFAULT_THREADS = Runtime.getRuntime().availableProcessors();

    /**
     * Override this to specify the number of threads for C++ execution, via
     * {@link org.nd4j.linalg.factory.Environment#setMaxMasterThreads(int)}
     * @return Number of threads to use for C++ op execution
     */
    public int numThreads() {
        return DEFAULT_THREADS;
    }

    /**
     * Override this method to set the default timeout for methods in the test class
     */
    public long getTimeoutMilliseconds() {
        return 90_000;
    }

    /**
     * Override this to set the profiling mode for the tests defined in the child class
     */
    public OpExecutioner.ProfilingMode getProfilingMode() {
        return OpExecutioner.ProfilingMode.SCOPE_PANIC;
    }

    /**
     * Override this to set the datatype of the tests defined in the child class
     */
    public DataType getDataType() {
        return DataType.DOUBLE;
    }

    public DataType getDefaultFPDataType() {
        return getDataType();
    }

    protected static Boolean integrationTest;

    /**
     * @return True if integration tests maven profile is enabled, false otherwise.
     */
    public static boolean isIntegrationTests() {
        if (integrationTest == null) {
            String prop = System.getenv("DL4J_INTEGRATION_TESTS");
            integrationTest = Boolean.parseBoolean(prop);
        }
        return integrationTest;
    }

    /**
     * Call this as the first line of a test in order to skip that test, only when the integration tests maven profile is not enabled.
     * This can be used to dynamically skip integration tests when the integration test profile is not enabled.
     * Note that the integration test profile is not enabled by default - "integration-tests" profile
     */
    public static void skipUnlessIntegrationTests() {
        assumeTrue(isIntegrationTests(), "Skipping integration test - integration profile is not enabled");
    }

}
