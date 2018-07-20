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

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;

/**
 * Created by susaneraly on 12/14/17.
 * Run all the tests found in a subdir - for debug
 * eg. run all tests under tf_graphs/examples/norm_tests
 */
@RunWith(Parameterized.class)
@Slf4j
public class TFGraphTestSubDir {
    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    public static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
    //public static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.LIBND4J;
    //public static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.JUST_PRINT;
    private static final String[] SKIP_ARR = new String[] {
            //"norm_11",
            "one_hot"
    };
    public static final Set<String> SKIP_SET = new HashSet<>(Arrays.asList(SKIP_ARR));
    public static String modelDir = "tf_graphs/examples/simple_run";

    @Parameterized.Parameters
    public static Collection<Object[]> data() throws IOException {
        return TFGraphTestAllHelper.fetchTestParams(modelDir,EXECUTE_WITH);
    }

    public TFGraphTestSubDir(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
    }

    @Test
    public void test() throws Exception {
        Nd4j.create(1);
        if (SKIP_SET.contains(modelName)) {
            log.info("\n\tSKIPPED MODEL: " + modelName);
            return;
        }
        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, modelDir, EXECUTE_WITH);
        TFGraphTestAllHelper.checkIntermediate(inputs,modelName,modelDir,EXECUTE_WITH);
    }
}
