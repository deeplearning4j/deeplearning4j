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

/**
 * <h1>libnd4j Native C++ Test Integration</h1>
 *
 * <p>This package provides integration between libnd4j's Google Test-based C++ test suite
 * and Java JUnit 5 tests, enabling unified test execution and reporting.</p>
 *
 * <h2>Overview</h2>
 * <p>The integration allows running libnd4j C++ tests alongside Java tests using Maven,
 * with proper XML reporting compatible with CI/CD systems like Jenkins.</p>
 *
 * <h2>Key Classes</h2>
 * <ul>
 *   <li>{@link org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jNativeTestRunner} -
 *       Main JUnit 5 test runner using DynamicTest for test discovery</li>
 *   <li>{@link org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests} -
 *       Categorized test runner for running specific test suites</li>
 *   <li>{@link org.eclipse.deeplearning4j.nd4j.libnd4j.Libnd4jTestHelper} -
 *       Utility class for discovering and running native tests</li>
 *   <li>{@link org.eclipse.deeplearning4j.nd4j.libnd4j.GTestResultParser} -
 *       Parser for Google Test XML output format</li>
 *   <li>{@link org.eclipse.deeplearning4j.nd4j.libnd4j.NativeBuildHelper} -
 *       Helper for building libnd4j with CMake</li>
 * </ul>
 *
 * <h2>Configuration</h2>
 * <p>Configure via system properties or Maven properties:</p>
 * <pre>
 * org.nd4j.libnd4j.home         - Path to libnd4j directory
 * org.nd4j.libnd4j.chip         - Target chip (cpu/cuda), default: cpu
 * org.nd4j.libnd4j.test.filter  - GTest filter pattern, default: *
 * org.nd4j.libnd4j.test.timeout - Timeout in minutes, default: 30
 * org.nd4j.libnd4j.test.enabled - Enable/disable native tests, default: true
 * org.nd4j.libnd4j.autobuild    - Auto-build if executable not found, default: false
 * </pre>
 *
 * <h2>Usage Examples</h2>
 *
 * <h3>Run all native tests via Maven:</h3>
 * <pre>
 * mvn test -Dtest=LibNd4jNativeTestRunner
 * </pre>
 *
 * <h3>Run specific test suite:</h3>
 * <pre>
 * mvn test -Dtest=LibNd4jNativeTestRunner -Dorg.nd4j.libnd4j.test.filter=ArrayOptionsTests
 * </pre>
 *
 * <h3>Run native tests profile:</h3>
 * <pre>
 * mvn test -Pnative-tests
 * </pre>
 *
 * <h3>Run with CUDA:</h3>
 * <pre>
 * mvn test -Pnative-tests-cuda
 * </pre>
 *
 * <h3>Run categorized tests:</h3>
 * <pre>
 * mvn test -Dtest=LibNd4jCategoryTests
 * </pre>
 *
 * <h3>Run specific category:</h3>
 * <pre>
 * mvn test -Dtest="LibNd4jCategoryTests\$DeclarableOpsTests"
 * </pre>
 *
 * <h2>Test Categories</h2>
 * <p>Tests are organized into categories for selective execution:</p>
 * <ul>
 *   <li><b>Declarable Operations</b> - DeclarableOpsTests1-19</li>
 *   <li><b>Array Tests</b> - NDArray, ArrayOptions, MultiDataType</li>
 *   <li><b>Convolution Tests</b> - ConvolutionTests1, ConvolutionTests2</li>
 *   <li><b>Attention &amp; NLP</b> - AttentionTests, NlpTests</li>
 *   <li><b>Helper Tests</b> - HelpersTests, ShapeTests, etc.</li>
 *   <li><b>Memory Tests</b> - WorkspaceTests, MemoryUtilsTests</li>
 *   <li><b>Graph Tests</b> - GraphTests, FlatBuffersTest</li>
 *   <li><b>Broadcast Tests</b> - BroadcastableOpsTests</li>
 *   <li><b>Legacy Tests</b> - LegacyOpsTests, ParityOpsTests</li>
 *   <li><b>RNG Tests</b> - RNGTests, GraphRandomGeneratorTests</li>
 *   <li><b>Backprop Tests</b> - BackpropTests</li>
 * </ul>
 *
 * <h2>Prerequisites</h2>
 * <p>Before running native tests:</p>
 * <ol>
 *   <li>Build libnd4j with tests enabled:
 *       <pre>cd libnd4j &amp;&amp; ./buildnativeoperations.sh -t</pre>
 *       or
 *       <pre>cmake -B blasbuild/cpu -S . -DBUILD_TESTS=ON -DSD_CPU=ON
 * cmake --build blasbuild/cpu --target runtests</pre>
 *   </li>
 *   <li>Ensure the runtests executable is at:
 *       <pre>libnd4j/blasbuild/cpu/tests_cpu/layers_tests/runtests</pre>
 *   </li>
 * </ol>
 *
 * <h2>CI/CD Integration</h2>
 * <p>Test results are output in JUnit XML format for CI integration.
 * Results are written to temporary directories and can be aggregated with Java test results.</p>
 *
 * @see org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jNativeTestRunner
 * @see org.eclipse.deeplearning4j.nd4j.libnd4j.LibNd4jCategoryTests
 * @see org.eclipse.deeplearning4j.nd4j.libnd4j.Libnd4jTestHelper
 */
package org.eclipse.deeplearning4j.nd4j.libnd4j;
