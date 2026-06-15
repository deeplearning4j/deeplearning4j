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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.annotation.*;

/**
 * Annotation to mark a test for execution on specific backends/helpers.
 *
 * Usage:
 * <pre>
 * {@code
 * @BackendTest(helpers = {"onednn", "cudnn"})
 * public void testConvolution() {
 *     // This test will run on OneDNN and cuDNN helpers
 * }
 *
 * @BackendTest(backends = {Backend.CUDA})
 * public void testCudaSpecific() {
 *     // This test only runs on CUDA backend
 * }
 *
 * @BackendTest(requiresCapability = "float16")
 * public void testFloat16() {
 *     // This test requires float16 support
 * }
 * }
 * </pre>
 *
 * @author Adam Gibson
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Tag("multi-backend")
@ExtendWith(MultiBackendTestExtension.class)
public @interface BackendTest {

    /**
     * Specific helpers to run this test on.
     * If empty, runs on all enabled helpers.
     *
     * Valid values: "cpu", "onednn", "cudnn", "armcompute", "mps", "accelerate", "mlir", "miopen"
     */
    String[] helpers() default {};

    /**
     * Backends to run this test on.
     * If empty, runs on the current backend.
     */
    MultiBackendTestConfiguration.Backend[] backends() default {};

    /**
     * Helpers to exclude from testing.
     */
    String[] excludeHelpers() default {};

    /**
     * Required capability for the test.
     * Test will only run on helpers that support this capability.
     *
     * Valid values: "float16", "bfloat16", "int8", "convolution", "rnn", "attention", "batchnorm"
     */
    String requiresCapability() default "";

    /**
     * Minimum memory required (in MB) for this test.
     * Useful for excluding memory-intensive tests on smaller GPUs.
     */
    long minMemoryMB() default 0;

    /**
     * Whether to run this test in parallel across helpers.
     * Default is false (sequential execution).
     */
    boolean parallel() default false;

    /**
     * Description of what this test validates across backends.
     */
    String description() default "";
}
