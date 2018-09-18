/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

import org.junit.After;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

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

    public static String[] modelNames = new String[]{
            "matrix_determinant/rank2_5,5"

    };

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    //change this to SAMEDIFF for samediff
    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
//    public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    // public static TFGraphTestAllHelper.ExecuteWith executeWith = TFGraphTestAllHelper.ExecuteWith.JUST_PRINT;

    public static final String MODEL_DIR = "tf_graphs/examples";
    public static final String MODEL_FILENAME = "frozen_model.pb";

    private String modelName;

    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        List<Object[]> modelNamesParams = new ArrayList<>();
        for (int i = 0; i < modelNames.length; i++) {
            Object[] currentParams = new String[]{modelNames[i]};
            modelNamesParams.add(currentParams);
        }
        return modelNamesParams;
    }

    public TFGraphTestList(String modelName) {
        this.modelName = modelName;
    }

    @Test
    public void testOutputOnly() throws IOException {
        File dir = testDir.newFolder();
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, MODEL_DIR, dir);
        Map<String, INDArray> predictions = TFGraphTestAllHelper.outputVars(modelName, MODEL_DIR, dir);
        Double precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, MODEL_DIR, MODEL_FILENAME, executeWith, TFGraphTestAllHelper.LOADER, precisionOverride);
    }

    @Test
    public void testAlsoIntermediate() throws IOException {
        File dir = testDir.newFolder();
        Map<String, INDArray> inputs = TFGraphTestAllHelper.inputVars(modelName, MODEL_DIR, dir);
        TFGraphTestAllHelper.checkIntermediate(inputs, modelName, MODEL_DIR, MODEL_FILENAME, executeWith, dir);

    }
}
