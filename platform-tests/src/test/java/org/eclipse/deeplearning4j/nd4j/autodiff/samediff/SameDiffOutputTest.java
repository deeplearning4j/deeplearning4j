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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.custom.Invoke;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


@NativeTag
@Tag(TagNames.SAMEDIFF)
public class SameDiffOutputTest extends BaseNd4jTestWithBackends {


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInvoke(Nd4jBackend backend) {
        SameDiff sameDiff = SameDiff.create();
        SameDiff subGraph = SameDiff.create();
        sameDiff.putSubFunction("add",subGraph);
        SDVariable inputOne = subGraph.placeHolder("input1",DataType.DOUBLE,2,2);
        SDVariable inputTwo = subGraph.placeHolder("input2",DataType.DOUBLE,2,2);
        SDVariable inputOneParent = sameDiff.placeHolder("input1",DataType.DOUBLE,2,2);
        SDVariable inputTwoParent = sameDiff.placeHolder("input2",DataType.DOUBLE,2,2);
        subGraph.math().add("add",inputOne,inputTwo);
        Invoke.InvokeParams invokeParams = Invoke.InvokeParams.builder()
                .functionName("add")
                .inputVarNames(new String[]{"input1","input2"})
                .outputVarNames(new String[]{"add"})
                .subGraphInputVarNames(new String[]{"input1","input2"})
                .subGraphOutputVarNames(new String[]{"add"})
                .inputs(new SDVariable[]{inputOneParent,inputTwoParent})
                .build();
        sameDiff.invoke(invokeParams);
        Map<String,INDArray> inputs = new LinkedHashMap<>();
        inputs.put("input1",Nd4j.ones(2,2));
        inputs.put("input2",Nd4j.ones(2,2).addi(1));
        Map<String, INDArray> stringINDArrayMap = sameDiff.output(inputs,"add");
        assertEquals(Nd4j.valueArrayOf(new int[]{2,2},3.0),stringINDArrayMap.get("add"));

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void outputTest(Nd4jBackend backend) {
        DataSet data = new DataSet(Nd4j.zeros(10, 10), Nd4j.zeros(10, 10));
        SameDiff sd = SameDiff.create();

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, 10, 10);
        SDVariable out = in.add("out", 2);

        TrainingConfig conf = new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(new Sgd(3e-1))
                .dataSetFeatureMapping("input")
                .dataSetLabelMapping()
                .build();

        sd.setTrainingConfig(conf);

        INDArray output = sd.output(data, "out").get("out");

        assertTrue(output.equalsWithEps(
                Nd4j.zeros(10, 10).add(2).castTo(DataType.FLOAT),
                0.0001),"output != input + 2");
    }

    @Override
    public char ordering() {
        return 'c';
    }

}
