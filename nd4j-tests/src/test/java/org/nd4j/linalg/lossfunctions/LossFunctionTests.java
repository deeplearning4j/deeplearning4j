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
 *
 */

package org.nd4j.linalg.lossfunctions;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer.Type;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Testing loss function
 *
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public  class LossFunctionTests extends BaseNd4jTest {


    public LossFunctionTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testCreateLossFunction() {
        LossFunction l = Nd4j.getOpFactory().createLossFunction(new TestLossFunction().name(),Nd4j.create(1),Nd4j.create(1));
        assertEquals(l.getClass(),TestLossFunction.class);
    }

    @Test
    public void testRMseXent() {
        INDArray in = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        INDArray out = Nd4j.create(new double[][]{{5, 6}, {7, 8}});
        double diff = LossFunctions.score(in, LossFunctions.LossFunction.RMSE_XENT, out, 0, false);
        assertEquals(getFailureMessage(),8, diff, 1e-1);
    }

    @Test
    public void testMcXent() {
        INDArray in = Nd4j.create(new float[][]{{1, 2}, {3, 4}});
        INDArray out = Nd4j.create(new float[][]{{5, 6}, {7, 8}});
        LossFunctions.score(in, LossFunctions.LossFunction.MCXENT, out, 0, false);
    }

    @Test
    public void testNegativeLogLikelihood() {
        Nd4j.dtype = Type.DOUBLE;
        INDArray softmax = Nd4j.create(new double[][]{{0.6, 0.4}, {0.7, 0.3}});
        INDArray trueLabels = Nd4j.create(new double[][]{{1, 0}, {0, 1}});
        double score = LossFunctions.score(trueLabels, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, softmax, 0, false);
        assertEquals(getFailureMessage(),0.8573992252349854, score, 1e-1);


        INDArray softmax2 = Nd4j.create(new double[][]{{0.33, 0.33, 0.33}, {0.33, 0.33, 0.33}});
        INDArray trueLabels2 = Nd4j.create(new double[][]{{1, 0, 0}, {1, 0, 0}});
        double score2 = LossFunctions.score(trueLabels2, LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD, softmax2, 0, false);
        assertEquals(getFailureMessage(),1.1086626052856445, score2, 1e-1);

    }

    @Override
    public char ordering() {
        return 'f';
    }
}
