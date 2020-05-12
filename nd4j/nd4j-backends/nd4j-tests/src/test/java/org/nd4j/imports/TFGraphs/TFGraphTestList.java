/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.imports.TFGraphs;

import org.junit.*;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * TFGraphTestAll* will run all the checked in TF graphs and
 * compare outputs in nd4j to those generated and checked in from TF.
 * <p>
 * This file is to run a single graph or a list of graphs to aid in debug.
 * Simply change the modelNames String[] to correspond to the directory name the graph lives in
 * - eg. to run the graph for 'bias_add' i.e checked in under tf_graphs/examples/bias_add
 * <p>
 */
@RunWith(Parameterized.class)
@Ignore
public class TFGraphTestList {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    //Only enable this for debugging, and leave it disabled for normal testing and CI - it prints all arrays for every execution step
    //Implemented internally using ExecPrintListener
    public static final boolean printArraysDebugging = false;

    public static String[] modelNames = new String[]{
            "emptyArrayTests/identity_n/rank1"
    };

    @After
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

    @BeforeClass
    public static void beforeClass(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    private String modelName;
    private TestCase testCase;

    @Parameterized.Parameters(name="{0}")
    public static Collection<Object[]> data() throws Exception {

        List<Object[]> out = new ArrayList<>(modelNames.length);
        for(int i=0; i<modelNames.length; i++ ) {
            String s = modelNames[i];
            TestCase t = TFGraphUtil.getTestCase(TFGraphTestAllSameDiff.BASE_DIR, s);


            out.add(new Object[]{s, t});
        }
        return out;
    }

    public TFGraphTestList(String modelName, TestCase tc) {
        this.modelName = modelName;
        this.testCase = tc;
    }

    @Test
    public void testOutputOnly() throws IOException {
        Map<String,INDArray> inputs = TFGraphUtil.loadInputs(testCase);
        Map<String,INDArray> predictions = TFGraphUtil.loadPredictions(testCase);

        Pair<Double,Double> precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        Double maxRE = (precisionOverride == null ? null : precisionOverride.getFirst());
        Double minAbs = (precisionOverride == null ? null : precisionOverride.getSecond());

        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, MODEL_DIR, MODEL_FILENAME, executeWith,
                TFGraphTestAllHelper.LOADER, maxRE, minAbs, printArraysDebugging);
    }

    @Test @Ignore
    public void testAlsoIntermediate() throws IOException {
        //Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);
        File dir = testDir.newFolder();
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, MODEL_DIR, dir);
        TFGraphTestAllHelper.checkIntermediate(inputs, modelName, MODEL_DIR, MODEL_FILENAME, executeWith, dir, printArraysDebugging);
    }
}
