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

import lombok.val;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class FailingSameDiffTests extends BaseNd4jTestWithBackends {


    @Override
    public char ordering(){
        return 'c';
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEyeShape(Nd4jBackend backend) {
        val dco = DynamicCustomOp.builder("eye")
                .addIntegerArguments(3,3)
                .build();

        val list = Nd4j.getExecutioner().calculateOutputShape(dco);
        assertEquals(1, list.size());   //Fails here - empty list
        assertArrayEquals(new long[]{3,3}, Shape.shape(list.get(0).asLong()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExecutionDifferentShapesTransform(Nd4jBackend backend){
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4));

        SDVariable tanh = sd.math().tanh(in);
        INDArray exp = Transforms.tanh(in.getArr(), true);

        INDArray out = tanh.eval();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1,20,20, DataType.DOUBLE).reshape(5,4));
        INDArray out2 = tanh.eval();
        assertArrayEquals(new long[]{5,4}, out2.shape());

        exp = Transforms.tanh(in.getArr(), true);
        assertEquals(exp, out2);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDropout(Nd4jBackend backend) {
        SameDiff sd = SameDiff.create();
        double p = 0.5;
        INDArray ia = Nd4j.create(new long[]{2, 2});

        SDVariable input = sd.var("input", ia);

        SDVariable res = sd.nn().dropout(input, false,p);
        Map<String, INDArray> output = sd.outputAll(Collections.emptyMap());
        assertTrue(!output.isEmpty());

       // assertArrayEquals(new long[]{2, 2}, res.eval().shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testExecutionDifferentShapesDynamicCustom(Nd4jBackend backend) {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4));
        SDVariable w = sd.var("w", Nd4j.linspace(1,20,20, DataType.DOUBLE).reshape(4,5));
        SDVariable b = sd.var("b", Nd4j.linspace(1,5,5, DataType.DOUBLE).reshape(1,5));

        SDVariable mmul = sd.mmul(in,w).add(b);
        INDArray exp = in.getArr().mmul(w.getArr()).addiRowVector(b.getArr());

        INDArray out = mmul.eval();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1,20,20, DataType.DOUBLE).reshape(5,4));
        INDArray out2 = mmul.eval();
        assertArrayEquals(new long[]{5,5}, out2.shape());

        exp = in.getArr().mmul(w.getArr()).addiRowVector(b.getArr());
        assertEquals(exp, out2);

        //Generate gradient function, and exec
        SDVariable loss = mmul.std(true);
        sd.calculateGradients(Collections.emptyMap(), sd.getVariables().keySet());

        in.setArray(Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4));
        out2 = mmul.eval();
        assertArrayEquals(new long[]{3,5}, out2.shape());
    }

}
