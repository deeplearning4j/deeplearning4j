package org.eclipse.deeplearning4j.frameworkimport.tensorflow.models;

/*
 *  ******************************************************************************
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


import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.frameworkimport.tensorflow.TFGraphTestAllHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.profiler.ProfilerConfig;

import java.io.File;
import java.util.*;

import static org.eclipse.deeplearning4j.frameworkimport.tensorflow.models.TestTFGraphAllSameDiffPartitionedBase.*;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

@Slf4j
class TestRunner {
    private final List<String> debugModeRegexes;

    public TestRunner(List<String> debugModeRegexes) {
        this.debugModeRegexes = debugModeRegexes;

    }

    public void runTest(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, File localTestDir) throws Exception {
        for (String s : IGNORE_REGEXES) {
            if (modelName.matches(s) || TFGraphTestAllHelper.failFastStop()) {
                log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, s);
                assumeFalse(true);
            }
        }


        Pair<Double, Double> precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        Double maxRE = (precisionOverride == null ? null : precisionOverride.getFirst());
        Double minAbs = (precisionOverride == null ? null : precisionOverride.getSecond());

        boolean verboseDebugMode = true;
        if (debugModeRegexes != null) {
            for (String regex : debugModeRegexes) {
                if (modelName.matches(regex)) {
                    verboseDebugMode = true;
                    break;
                }
            }
        }

        try {
            Nd4j.getEnvironment().setDeletePrimary(false);
            Nd4j.getEnvironment().setDeleteSpecial(false);
            Nd4j.getExecutioner().enableDebugMode(true);
            Nd4j.getExecutioner().enableVerboseMode(true);
            TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, new TFGraphTestAllHelper.DefaultGraphLoader(inputs), maxRE, minAbs, verboseDebugMode);
        } catch (Throwable t) {
            log.error("ERROR Executing test: {} - input keys {}", modelName, (inputs == null ? null : inputs.keySet()), t);
            throw t;
        }
    }
}

