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

package org.arbiter.eval;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.FeatureUtil;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 12/22/14.
 */
public class EvalTest {

    @Test
    public void testEval() {
        Evaluation eval = new Evaluation();
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0,5);
        INDArray test = FeatureUtil.toOutcomeVector(0,5);
        eval.eval(trueOutcome,test);
        assertEquals(1.0,eval.f1(),1e-1);
        assertEquals(1,eval.classCount(0));

        INDArray one = FeatureUtil.toOutcomeVector(1,5);
        INDArray miss = FeatureUtil.toOutcomeVector(0,5);
        eval.eval(one,miss);
        assertTrue(eval.f1() < 1);
        assertEquals(1,eval.classCount(0));
        assertEquals(1,eval.classCount(1));

        assertEquals(2.0,eval.positive(),1e-1);
        assertEquals(1.0,eval.negative(),1e-1);



    }


}
