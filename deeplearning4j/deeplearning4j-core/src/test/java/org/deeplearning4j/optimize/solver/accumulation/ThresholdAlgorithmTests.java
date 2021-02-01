/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.optimize.solver.accumulation;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithm;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithmReducer;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;

import static org.junit.Assert.assertEquals;

public class ThresholdAlgorithmTests extends BaseDL4JTest {

    @Test
    public void testAdaptiveThresholdAlgorithm() throws Exception {

        double initialThreshold = 1e-3;
        double minTargetSparsity = 1e-3;
        double maxTargetSparsity = 1e-2;
        double decayRate = 0.95;
        ThresholdAlgorithm ta = new AdaptiveThresholdAlgorithm(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate);

        //First call: expect last threshold, last was dense, last sparsity etc to be null
        INDArray update = Nd4j.rand(new long[]{1,100});
        update.muli(2).subi(1); //-1 to 1
        update.muli(initialThreshold * 2);      //-2*initialThreshold to 2*initialThreshold

        double t1 = ta.calculateThreshold(0, 0, null, null, null, update);
        assertEquals(initialThreshold, t1, 0.0);

        //Second call: assume first encoding was dense using initial threshold -> increase threshold (reduce sparsity ratio -> more sparse)
        double t2 = ta.calculateThreshold(1, 0, initialThreshold, true, null, update);
        double expT2 = (1.0 / decayRate) * initialThreshold;
        assertEquals(expT2, t2, 1e-6);

        //Third call: assume second encoding was sparse, but greater than max sparsity target -> increase threshold (reduce sparsity ratio -> more sparse)
        double t3 = ta.calculateThreshold(2, 0, t2, false, 1e-1, update);
        double expT3 = (1.0 / decayRate) * t2;
        assertEquals(expT3, t3, 1e-6);

        //Fourth call: assume third encoding was sparse, but smaller than min sparsity target -> decrease threshold (increase sparsity ratio -> less sparse)
        double t4 = ta.calculateThreshold(3, 0, t3, false, 1e-4, update);
        double expT4 = decayRate * t3;
        assertEquals(expT4, t4, 1e-6);

        //Check that the last threshold is set:
        Field f = AdaptiveThresholdAlgorithm.class.getDeclaredField("lastThreshold");
        f.setAccessible(true);
        double fValue = (Double)f.get(ta);
        assertEquals(t4, fValue, 0.0);

        //Check combining:
        AdaptiveThresholdAlgorithm ta2 = (AdaptiveThresholdAlgorithm) ta.clone();
        assertEquals(ta, ta2);

        ThresholdAlgorithmReducer reducer = ta.newReducer();
        reducer.add(ta);
        reducer.add(ta2);

        ThresholdAlgorithm reduced = reducer.getFinalResult();
        assertEquals(reduced, ta);

        //Check combining with unused:
        reducer.add(new AdaptiveThresholdAlgorithm(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate));
        reduced = reducer.getFinalResult();
        assertEquals(reduced, ta);

        //Check "first iteration in second epoch" uses the stored threshold, not the passed in one
        //Should re-use last sparsity ratio of 1e-4 -> decrease threshold
        double t5 = reduced.calculateThreshold(5, 1, null, null, null, update);
        double expT5 = decayRate * t4;
        assertEquals(expT5, t5, 1e-6);


        //Check combining with different values:
        ThresholdAlgorithm taA = new AdaptiveThresholdAlgorithm(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate);
        ThresholdAlgorithm taB = new AdaptiveThresholdAlgorithm(initialThreshold, minTargetSparsity, maxTargetSparsity, decayRate);

        f.set(taA, 1e-4);
        f.set(taB, 5e-4);

        ThresholdAlgorithmReducer r2 = taA.newReducer();
        r2.add(taA);

        ThresholdAlgorithmReducer r3 = taB.newReducer();
        r3.add(taB);

        r2.merge(r3);

        reduced = r2.getFinalResult();

        fValue = (Double)f.get(reduced);
        assertEquals((1e-4 + 5e-4)/2.0, fValue, 1e-10);
    }

}
