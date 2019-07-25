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

package org.nd4j.autodiff.samediff;

import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.impl.DefaultSameDiffConditional;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@Ignore("AB 2019/05/21 - JVM Crash on ppc64 - Issue #7657")
public class FailingSameDiffTests extends BaseNd4jTest {

    public FailingSameDiffTests(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    public void testEye(){
        //OpValidationSuite.ignoreFailing();
        INDArray arr = Nd4j.create(new double[]{1, 0, 0, 0, 1, 0}, new int[]{2, 3});
        List<INDArray> stack = new ArrayList<>();
        for(int i=0; i< 25; i++){
            stack.add(arr);
        }
        INDArray expOut = Nd4j.pile(stack).reshape(5, 5, 2, 3);

        SameDiff sd = SameDiff.create();
        SDVariable result = sd.math().eye(2, 3, DataType.DOUBLE, 5, 5);

        assertEquals(expOut, result.eval());
    }

    @Test
    public void testEyeShape(){
        val dco = DynamicCustomOp.builder("eye")
                .addIntegerArguments(3,3)
                //.addIntegerArguments(-99,3,3) //Also fails
                .build();

        val list = Nd4j.getExecutioner().calculateOutputShape(dco);
        assertEquals(1, list.size());   //Fails here - empty list
        assertArrayEquals(new long[]{3,3}, list.get(0).getShape());
    }

    @Test(timeout = 10000L)
    public void testWhileLoop() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new DefaultSameDiffConditional(), new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].add(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        sameDiff.exec(Collections.emptyMap());
    }

    @Test(timeout = 10000L)
    public void testWhileBackwards() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new DefaultSameDiffConditional(), new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].add(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        sameDiff.execBackwards(Collections.emptyMap());
        SameDiff exec = sameDiff.getFunction("grad");
    }

    @Test(timeout = 10000L)
    public void testWhileLoop2() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new DefaultSameDiffConditional(), new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].add(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        sameDiff.exec(Collections.emptyMap(), sameDiff.outputs());
    }

    @Test
    public void testExecutionDifferentShapesTransform(){
        OpValidationSuite.ignoreFailing();
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4));

        SDVariable tanh = sd.math().tanh(in);
        INDArray exp = Transforms.tanh(in.getArr(), true);

        INDArray out = sd.execAndEndResult();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1,20,20, DataType.DOUBLE).reshape(5,4));
        INDArray out2 = sd.execAndEndResult();
        assertArrayEquals(new long[]{5,4}, out2.shape());

        exp = Transforms.tanh(in.getArr(), true);
        assertEquals(exp, out2);
    }

    @Test
    public void testDropout() {
        OpValidationSuite.ignoreFailing();
        SameDiff sd = SameDiff.create();
        double p = 0.5;
        INDArray ia = Nd4j.create(new long[]{2, 2});

        SDVariable input = sd.var("input", ia);

        SDVariable res = sd.nn().dropout(input, p);
        assertArrayEquals(new long[]{2, 2}, res.getShape());
    }

    @Test
    public void testExecutionDifferentShapesDynamicCustom(){
        OpValidationSuite.ignoreFailing();

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4));
        SDVariable w = sd.var("w", Nd4j.linspace(1,20,20, DataType.DOUBLE).reshape(4,5));
        SDVariable b = sd.var("b", Nd4j.linspace(1,5,5, DataType.DOUBLE).reshape(1,5));

        SDVariable mmul = sd.mmul(in,w).add(b);
        INDArray exp = in.getArr().mmul(w.getArr()).addiRowVector(b.getArr());

        INDArray out = sd.execAndEndResult();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1,20,20, DataType.DOUBLE).reshape(5,4));
        INDArray out2 = sd.execAndEndResult();
        assertArrayEquals(new long[]{5,5}, out2.shape());

        exp = in.getArr().mmul(w.getArr()).addiRowVector(b.getArr());
        assertEquals(exp, out2);

        //Generate gradient function, and exec
        SDVariable loss = mmul.std(true);
        sd.execBackwards(Collections.emptyMap());

        in.setArray(Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape(3,4));
        sd.execAndEndResult();
        out2 = mmul.getArr();
        assertArrayEquals(new long[]{3,5}, out2.shape());
    }

}
