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
import org.junit.Test;
import org.nd4j.autodiff.OpValidationSuite;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayV3;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

public class FailingSameDiffTests {

    @Test
    public void testEye(){
        OpValidationSuite.ignoreFailing();
        INDArray arr = Nd4j.create(new double[]{1, 0, 0, 0, 1, 0}, new int[]{2, 3});
        List<INDArray> stack = new ArrayList<>();
        for(int i=0; i< 25; i++){
            stack.add(arr);
        }
        INDArray expOut = Nd4j.pile(stack).reshape(5, 5, 2, 3);

        SameDiff sd = SameDiff.create();
        SDVariable result = sd.eye(2, 3, 5, 5);

        assertEquals(expOut, result.eval());
    }

    @Test(timeout = 10000L)
    public void testWhileLoop() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();
        assertFalse(exec.getRight().isEmpty());
        While function = (While) exec.getRight().get(exec.getRight().size() - 1);
        assumeNotNull(function.getOutputVars());
        assertEquals(1, function.getNumLooped());
        sameDiff.toString();
    }

    @Test(timeout = 10000L)
    public void testWhileBackwards() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].addi(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        sameDiff.execBackwards();
        SameDiff exec = sameDiff.getFunction("grad");
        System.out.println(exec);
    }

    @Test(timeout = 10000L)
    public void testTensorArray4(){
        OpValidationSuite.ignoreFailing();
        SameDiff sd = SameDiff.create();
        TensorArrayV3 ta = sd.tensorArray();

        // while loop
        val predicate = new SameDiff.DefaultSameDiffConditional();
        val cond = new SameDiff.SameDiffFunctionDefinition(){
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{ret};
            }
        };
        val loop_body = new SameDiff.SameDiffFunctionDefinition(){
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                ta.write(variableInputs[0], variableInputs[2]);
                SDVariable ret1 = variableInputs[0].addi(1);
                SDVariable ret2 = variableInputs[1];
                SDVariable ret3 = variableInputs[2].addi(1);
                return new SDVariable[]{ret1, ret2, ret3};
            }
        };

        SDVariable loop_counter = sd.var(Nd4j.create(new double[]{0}));


        INDArray arr = Nd4j.create(new double[]{1, 2, 3, 4, 5});
        SDVariable initial_state = sd.var(arr);

        sd.whileStatement(predicate, cond, loop_body, new SDVariable[]{loop_counter, loop_counter.add(10), initial_state});


        // build expected output
        List<INDArray> arr_list = new ArrayList<>();
        for(int i=0; i<10; i++){
            arr_list.add(arr.add(i));
        }
        INDArray expOut = Nd4j.pile(arr_list);


        SDVariable result = ta.stack();
        assertEquals(expOut, result.eval());
    }

    @Test(timeout = 10000L)
    public void testWhileLoop2() {
        OpValidationSuite.ignoreFailing();
        SameDiff sameDiff = SameDiff.create();
        sameDiff.whileStatement(new SameDiff.DefaultSameDiffConditional(), new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable eqResult = sameDiff.neq(variableInputs[0], variableInputs[1]);
                return new SDVariable[]{eqResult};
            }
        }, new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                SDVariable ret = variableInputs[1].add(1.0);
                return new SDVariable[]{variableInputs[0], ret};
            }
        }, new SDVariable[]{
                sameDiff.one("one", new long[]{1, 1}),
                sameDiff.var("two", new long[]{1, 1}),

        });

        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec = sameDiff.exec();
        assertFalse(exec.getRight().isEmpty());
        While function = (While) exec.getRight().get(exec.getRight().size() - 1);
        assumeNotNull(function.getOutputVars());
        assertEquals(1, function.getNumLooped());
        sameDiff.toString();
    }

    @Test
    public void testExecutionDifferentShapesTransform(){
        OpValidationSuite.ignoreFailing();
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12).reshape(3,4));

        SDVariable tanh = sd.tanh(in);
        INDArray exp = Transforms.tanh(in.getArr(), true);

        INDArray out = sd.execAndEndResult();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1,20,20).reshape(5,4));
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

        SDVariable res = sd.dropout(input, p);
        assertArrayEquals(new long[]{2, 2}, res.getShape());
    }

    @Test
    public void testExecutionDifferentShapesDynamicCustom(){
        OpValidationSuite.ignoreFailing();

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.linspace(1,12,12).reshape(3,4));
        SDVariable w = sd.var("w", Nd4j.linspace(1,20,20).reshape(4,5));
        SDVariable b = sd.var("b", Nd4j.linspace(1,5,5).reshape(1,5));

        SDVariable mmul = sd.mmul(in,w).addi(b);
        INDArray exp = in.getArr().mmul(w.getArr()).addiRowVector(b.getArr());

        INDArray out = sd.execAndEndResult();
        assertEquals(exp, out);

        //Now, replace with minibatch 5:
        in.setArray(Nd4j.linspace(1,20,20).reshape(5,4));
        INDArray out2 = sd.execAndEndResult();
        assertArrayEquals(new long[]{5,5}, out2.shape());

        exp = in.getArr().mmul(w.getArr()).addiRowVector(b.getArr());
        assertEquals(exp, out2);

        //Generate gradient function, and exec
        SDVariable loss = mmul.std(true);
        sd.execBackwards();

        in.setArray(Nd4j.linspace(1,12,12).reshape(3,4));
        sd.execAndEndResult();
        out2 = mmul.getArr();
        assertArrayEquals(new long[]{3,5}, out2.shape());
    }

}
