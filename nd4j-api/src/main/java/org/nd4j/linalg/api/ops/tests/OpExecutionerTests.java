/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.linalg.api.ops.tests;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.exception.IllegalOpException;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.*;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 2/22/15.
 */
public abstract class OpExecutionerTests {


    @Test
    public void testExecutioner() throws IllegalOpException {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.ones(5);
        INDArray xDup = x.dup();
        INDArray solution = Nd4j.valueArrayOf(5,2.0);
        opExecutioner.exec(new AddOp(x,xDup,x));
        assertEquals(solution,x);
        Sum acc = new Sum(x.dup());
        opExecutioner.exec(acc);
        assertEquals(10.0,acc.currentResult().doubleValue(),1e-1);
        Prod prod = new Prod(x.dup());
        opExecutioner.exec(prod);
        assertEquals(32.0,prod.currentResult().doubleValue(),1e-1);
    }

    @Test
    public void testMaxMin() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray x = Nd4j.linspace(1,5,5);
        Max max = new Max(x);
        opExecutioner.exec(max);
        assertEquals(5,max.currentResult().doubleValue(),1e-1);
        Min min = new Min(x);
        assertEquals(1,min.currentResult().doubleValue(),1e-1);
        Mean mean = new Mean(x);
        opExecutioner.exec(mean);
        assertEquals(3.0,mean.currentResult().doubleValue(),1e-1);


    }

    @Test
    public void testSoftMax() {
        OpExecutioner opExecutioner = Nd4j.getExecutioner();
        INDArray arr = Nd4j.linspace(1,6,6).reshape(2,3);
        SoftMax softMax = new SoftMax(arr);
        opExecutioner.exec(softMax,0);
        for(int i = 0; i < arr.slices(); i++) {
            assertEquals(1.0,arr.slice(i,0).sum(Integer.MAX_VALUE).getDouble(0),1e-1);
        }
    }




}
