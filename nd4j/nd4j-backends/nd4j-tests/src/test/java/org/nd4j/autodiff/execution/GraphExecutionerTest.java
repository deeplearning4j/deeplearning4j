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

package org.nd4j.autodiff.execution;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Comparative tests for native executioner vs sequential execution
 * @author raver119@gmail.com
 */
@Slf4j
public class GraphExecutionerTest {
    protected static ExecutorConfiguration configVarSpace = ExecutorConfiguration.builder().outputMode(OutputMode.VARIABLE_SPACE).build();
    protected static ExecutorConfiguration configExplicit = ExecutorConfiguration.builder().outputMode(OutputMode.EXPLICIT).build();
    protected static ExecutorConfiguration configImplicit = ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build();

    @Before
    public void setUp() throws Exception {
        //
    }

    @Test
    @Ignore
    public void testConversion() throws Exception {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones",ones);
        SDVariable result = sdVariable.addi(1.0);
        SDVariable total = sameDiff.sum(result,Integer.MAX_VALUE);

        val executioner = new NativeGraphExecutioner();

        ByteBuffer buffer = executioner.convertToFlatBuffers(sameDiff, ExecutorConfiguration.builder().profilingMode(OpExecutioner.ProfilingMode.DISABLED).executionMode(ExecutionMode.SEQUENTIAL).outputMode(OutputMode.IMPLICIT).build());

        val offset = buffer.position();
        val array = buffer.array();

        try (val fos = new FileOutputStream("../../libnd4j/tests/resources/adam_sum.fb"); val dos = new DataOutputStream(fos)) {
            dos.write(array, offset, array.length - offset);
        }


        //INDArray[] res = executioner.executeGraph(sameDiff);
        //assertEquals(8.0, res[0].getDouble(0), 1e-5);
        /*
        INDArray output = null;
        for(int i = 0; i < 5; i++) {
            output = sameDiff.execAndEndResult(ops);
            System.out.println("Ones " + ones);
            System.out.println(output);
        }

        assertEquals(Nd4j.valueArrayOf(4,7),ones);
        assertEquals(28,output.getDouble(0),1e-1);
        */
    }


    /**
     * VarSpace should dump everything. 4 variables in our case
     * @throws Exception
     */
    @Test
    public void testEquality1() throws Exception {
        GraphExecutioner executionerA = new BasicGraphExecutioner();
        GraphExecutioner executionerB = new NativeGraphExecutioner();

        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones",ones);
        SDVariable scalarOne = sameDiff.var("scalar",Nd4j.scalar(1.0));
        SDVariable result = sdVariable.add(scalarOne);
        SDVariable total = sameDiff.sum(result,Integer.MAX_VALUE);

        log.info("TOTAL: {}; Id: {}", total.getVarName(), total);

        INDArray[] resB = executionerB.executeGraph(sameDiff, configVarSpace);

        //Variables: ones, scalar, result, total
        assertEquals(sameDiff.variables().size(), resB.length);
        assertEquals(Nd4j.ones(4), resB[0]);
        assertEquals(Nd4j.scalar(1), resB[1]);
        assertEquals(Nd4j.create(new float[]{2f, 2f, 2f, 2f}), resB[2]);
        assertEquals(Nd4j.scalar(8.0), resB[3]);
    }


    /**
     * Implicit should return tree edges. So, one variable
     * @throws Exception
     */
    @Test
    public void testEquality2() throws Exception {
        GraphExecutioner executionerA = new BasicGraphExecutioner();
        GraphExecutioner executionerB = new NativeGraphExecutioner();

        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones",ones);
        SDVariable scalarOne = sameDiff.var("add1",Nd4j.scalar(1.0));
        SDVariable result = sdVariable.add(scalarOne);
        SDVariable total = sameDiff.sum(result,Integer.MAX_VALUE);

//        log.info("ID: {}",sameDiff.getGraph().getVertex(1).getValue().getId());

        INDArray[] resB = executionerB.executeGraph(sameDiff, configImplicit);

        assertEquals(1, resB.length);
        assertEquals(Nd4j.scalar(8.0), resB[0]);

        //INDArray resA = executionerA.executeGraph(sameDiff)[0];

        //assertEquals(resA, resB);
    }


    @Test
    @Ignore
    public void testSums1() throws Exception {
        SameDiff sameDiff = SameDiff.create();
        INDArray ones = Nd4j.ones(4);
        SDVariable sdVariable = sameDiff.var("ones",ones);
        SDVariable result = sdVariable.addi(1.0);
        SDVariable total = sameDiff.sum(result,Integer.MAX_VALUE);

        val executioner = new NativeGraphExecutioner();

        INDArray[] res = executioner.executeGraph(sameDiff);
        assertEquals(8.0, res[0].getDouble(0), 1e-5);
    }
}