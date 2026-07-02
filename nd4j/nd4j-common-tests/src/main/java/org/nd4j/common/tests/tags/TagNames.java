/*
 *
 *  *  ******************************************************************************
 *  *  *
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  *  See the NOTICE file distributed with this work for additional
 *  *  *  information regarding copyright ownership.
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package org.nd4j.common.tests.tags;

public class TagNames {

    public final static String SAMEDIFF = "samediff"; //tests related to samediff
    public final static String RNG = "rng"; //tests related to RNG
    public final static String JAVA_ONLY = "java-only"; //tests with only pure java involved
    public final static String FILE_IO = "file-io"; // tests with file i/o
    public final static String DL4J_OLD_API = "dl4j-old-api"; //tests involving old dl4j api
    public final static String WORKSPACES = "workspaces"; //tests involving workspaces
    public final static String MULTI_THREADED = "multi-threaded"; //tests involving multi threading
    public final static String TRAINING = "training"; //tests related to training models
    public final static String LOSS_FUNCTIONS = "loss-functions"; //tests related to loss functions
    public final static String UI = "ui"; //ui related tests
    public final static String EVAL_METRICS = "model-eval-metrics"; //model evaluation metrics related
    public final static String CUSTOM_FUNCTIONALITY = "custom-functionality"; //tests related to custom ops, loss functions, layers
    public final static String JACKSON_SERDE = "jackson-serde"; //tests related to jackson serialization
    public final static String NDARRAY_INDEXING = "ndarray-indexing"; //tests related to ndarray slicing
    public final static String NDARRAY_SERDE = "ndarray-serde"; //tests related to ndarray serialization
    public final static String COMPRESSION = "compression"; //tests related to compression
    public final static String NDARRAY_ETL = "ndarray-etl"; //tests related to data preparation such as transforms and normalization
    public final static String MANUAL = "manual"; //tests related to running manually
    public final static String SPARK = "spark"; //tests related to apache spark
    public final static String DIST_SYSTEMS = "distributed-systems";
    public final static String SOLR = "solr";
    public final static String KERAS = "keras";
    public final static String PYTHON = "python";
    public final static String LONG_TEST = "long-running-test";
    public final static String NEEDS_VERIFY = "needs-verify"; //tests that need verification of issue
    public final static String LARGE_RESOURCES = "large-resources";
    public final static String DOWNLOADS = "downloads";
    public final static String TENSORFLOW = "tensorflow";
    public final static String ONNX = "onnx";

    // Test tiering tags - for CI pipeline selection
    public final static String SMOKE = "smoke"; //quick sanity checks (<30s total), safe for low-spec CI
    public final static String FULL_CI = "full-ci"; //broader validation (<5 min total), DSP lifecycle, evaluation, op basics
    // Tests without smoke or full-ci tags are implicitly long-running (nightly/weekly)

    // Accelerator / alternative-backend tags - selected by platform-tests pom profiles:
    // -Ptest-zluda -> zluda,rocm,amd-gpu | -Ptest-tpu -> tpu |
    // multi-backend-dual/-all -> multi-backend,multi-device[,backend-discovery]
    public final static String ZLUDA = "zluda"; //CUDA-on-AMD/Intel via ZLUDA (requires ZLUDA_PATH env)
    public final static String ROCM = "rocm"; //requires a ROCm-capable AMD GPU
    public final static String AMD_GPU = "amd-gpu"; //requires AMD GPU hardware
    public final static String TPU = "tpu"; //PJRT/libtpu backend tests (PJRT_PATH / TPU_LIBRARY_PATH)
    public final static String HEXAGON = "hexagon"; //Qualcomm Hexagon NPU (hexagon-mlir) backend tests
    public final static String METAL = "metal"; //Apple Metal/MLX backend tests (macOS arm64, -Ptest-metal)
    public final static String MULTI_BACKEND = "multi-backend"; //tests spanning >1 nd4j backend in one JVM
    public final static String MULTI_DEVICE = "multi-device"; //tests requiring >1 compute device
    public final static String BACKEND_DISCOVERY = "backend-discovery"; //Nd4jBackend SPI discovery/priority tests
}
