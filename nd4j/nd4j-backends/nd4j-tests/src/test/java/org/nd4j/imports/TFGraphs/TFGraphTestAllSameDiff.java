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
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.IOException;
import java.util.*;

/**
 * Created by susaneraly on 11/29/17.
 */
@Slf4j
@RunWith(Parameterized.class)
public class TFGraphTestAllSameDiff {
    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
    private static final String[] SKIP_ARR = new String[] {
            "deep_mnist",
            "deep_mnist_no_dropout",
            "ssd_mobilenet_v1_coco",
            "yolov2_608x608",
            "inception_v3_with_softmax",
            "conv_5" // this test runs, but we can't make it pass atm due to different RNG algorithms
    };
    public static final Set<String> SKIP_SET = new HashSet<>(Arrays.asList(SKIP_ARR));

    @BeforeClass
    public static void beforeClass() throws Exception {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @Before
    public void setup() {
        Nd4j.setDataType(DataBuffer.Type.FLOAT);
    }

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Parameterized.Parameters
    public static Collection<Object[]> data() throws IOException {
        return TFGraphTestAllHelper.fetchTestParams(EXECUTE_WITH);
    }

    public TFGraphTestAllSameDiff(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
    }

    @Test
    public void testOutputOnly() throws Exception {
        Nd4j.create(1);
        if (SKIP_SET.contains(modelName)) {
            log.info("\n\tSKIPPED MODEL: " + modelName);
            return;
        }
        Double precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);

        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, EXECUTE_WITH, precisionOverride);
        //TFGraphTestAllHelper.checkIntermediate(inputs, modelName, EXECUTE_WITH);
    }

}
