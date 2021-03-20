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
}
