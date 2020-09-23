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

package org.nd4j.linalg.lossfunctions;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;

import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 09/09/2016.
 */
public class LossFunctionTest extends BaseNd4jTest {

    public LossFunctionTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testClippingXENT() {

        ILossFunction l1 = new LossBinaryXENT(0);
        ILossFunction l2 = new LossBinaryXENT();

        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.create(3, 5), 0.5));
        INDArray preOut = Nd4j.valueArrayOf(3, 5, -1000.0);

        IActivation a = new ActivationSigmoid();

        double score1 = l1.computeScore(labels, preOut.dup(), a, null, false);
        assertTrue(Double.isNaN(score1));

        double score2 = l2.computeScore(labels, preOut.dup(), a, null, false);
        assertFalse(Double.isNaN(score2));

        INDArray grad1 = l1.computeGradient(labels, preOut.dup(), a, null);
        INDArray grad2 = l2.computeGradient(labels, preOut.dup(), a, null);

        MatchCondition c1 = new MatchCondition(grad1, Conditions.isNan());
        MatchCondition c2 = new MatchCondition(grad2, Conditions.isNan());
        int match1 = Nd4j.getExecutioner().exec(c1).getInt(0);
        int match2 = Nd4j.getExecutioner().exec(c2).getInt(0);

        assertTrue(match1 > 0);
        assertEquals(0, match2);
    }

    @Test
    public void testWeightedLossFunctionDTypes(){

        for(DataType activationsDt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
            for(DataType weightsDt : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
                for( boolean rank1W : new boolean[]{false, true}) {

                    INDArray preOut = Nd4j.rand(activationsDt, 2, 3);
                    INDArray l = Nd4j.rand(activationsDt, 2, 3);

                    INDArray w = Nd4j.createFromArray(1.0f, 2.0f, 3.0f).castTo(weightsDt);
                    if(!rank1W){
                        w = w.reshape(1, 3);
                    }

                    ILossFunction lf = null;
                    for (int i = 0; i < 10; i++) {
                        switch (i) {
                            case 0:
                                lf = new LossBinaryXENT(w);
                                break;
                            case 1:
                                lf = new LossL1(w);
                                break;
                            case 2:
                                lf = new LossL2(w);
                                break;
                            case 3:
                                lf = new LossMAE(w);
                                break;
                            case 4:
                                lf = new LossMAPE(w);
                                break;
                            case 5:
                                lf = new LossMCXENT(w);
                                break;
                            case 6:
                                lf = new LossMSE(w);
                                break;
                            case 7:
                                lf = new LossMSLE(w);
                                break;
                            case 8:
                                lf = new LossNegativeLogLikelihood(w);
                                break;
                            case 9:
                                lf = new LossSparseMCXENT(w);
                                l = Nd4j.createFromArray(1,2).reshape(2, 1).castTo(activationsDt);
                                break;
                            default:
                                throw new RuntimeException();
                        }
                    }

                    //Check score
                    lf.computeScore(l, preOut, new ActivationSoftmax(), null, true);

                    //Check backward
                    lf.computeGradient(l, preOut, new ActivationSoftmax(), null);
                }
            }
        }

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
