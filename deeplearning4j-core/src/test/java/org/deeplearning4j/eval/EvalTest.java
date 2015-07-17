/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.eval;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 12/22/14.
 */
public class EvalTest {

    @Test
    public void testEval() {
        int classNum = 5;
        Evaluation eval = new Evaluation(classNum);

        // Testing the edge case when some classes do not have true positive
        INDArray trueOutcome = FeatureUtil.toOutcomeVector(0, 5);
        INDArray predictedOutcome = FeatureUtil.toOutcomeVector(0, 5);
        eval.eval(trueOutcome, predictedOutcome);
        assertEquals(1, eval.classCount(0));
        assertEquals(1.0, eval.f1(), 1e-1);

        // Testing more than one sample. eval() does not reset the Evaluation instance
        INDArray trueOutcome2 = FeatureUtil.toOutcomeVector(1, 5);
        INDArray predictedOutcome2 = FeatureUtil.toOutcomeVector(0, 5);
        eval.eval(trueOutcome2, predictedOutcome2);
        // Verified with sklearn in Python
        // from sklearn.metrics import classification_report
        // classification_report(['a', 'a'], ['a', 'b'], labels=['a', 'b', 'c', 'd', 'e'])
        assertEquals(eval.f1(), 0.6, 1e-1);
        // The first entry is 0 label
        assertEquals(1, eval.classCount(0));
        // The first entry is 1 label
        assertEquals(1, eval.classCount(1));
        // Two positives since two entries
        assertEquals(2, eval.positive(), 0);
        // The rest are negative
        assertEquals(8, eval.negative(), 0);
        // 2 rows and only the first is correct
        assertEquals(eval.accuracy(), 0.5, 0);
    }
}
