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

package org.nd4j.imports.tfgraphs;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.io.TempDir;


import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;


@Disabled
public class TFGraphTestList {


    //Only enable this for debugging, and leave it disabled for normal testing and CI - it prints all arrays for every execution step
    //Implemented internally using ExecPrintListener
    public static final boolean printArraysDebugging = false;

    public static String[] modelNames = new String[]{
            "resize_nearest_neighbor/int32"
    };

    @AfterEach
    public void tearDown() {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    //change this to SAMEDIFF for samediff
    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
//    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    // public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.JUST_PRINT;

    public static final String MODEL_DIR = "tf_graphs/examples";
    public static final String MODEL_FILENAME = "frozen_model.pb";

    @BeforeAll
    public static void beforeClass() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    private String modelName;


    public static Stream<Arguments> data() {
        List<Object[]> modelNamesParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new String[]{modelNames[i]};
            modelNamesParams.add(currentParams);
        }
        return modelNamesParams.stream().map(Arguments::of);
    }


    @ParameterizedTest
    @MethodSource("#data")
    public void testOutputOnly(@TempDir Path testDir,String modelName) throws IOException {
        //Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);
        File dir = testDir.toFile();
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, MODEL_DIR, dir);
        Map<String, INDArray> predictions = TFGraphTestAllHelper.outputVars(modelName, MODEL_DIR, dir);
        Pair<Double,Double> precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        Double maxRE = (precisionOverride == null ? null : precisionOverride.getFirst());
        Double minAbs = (precisionOverride == null ? null : precisionOverride.getSecond());

        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, MODEL_DIR, MODEL_FILENAME, executeWith,
                TFGraphTestAllHelper.LOADER, maxRE, minAbs, printArraysDebugging);
    }

    @Test @Disabled
    @ParameterizedTest
    @MethodSource("#data")
    public void testAlsoIntermediate(@TempDir Path testDir,String modelName) throws IOException {
        //Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);
        File dir = testDir.toFile();
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, MODEL_DIR, dir);
        TFGraphTestAllHelper.checkIntermediate(inputs, modelName, MODEL_DIR, MODEL_FILENAME, executeWith, dir, printArraysDebugging);
    }
}
