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

package org.deeplearning4j.rl4j.network.ac;

import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 *
 * @author saudet
 */
public class ActorCriticTest {

    public static ActorCriticFactorySeparateStdDense.Configuration NET_CONF =
            new ActorCriticFactorySeparateStdDense.Configuration(
                    4,         //number of layers
                    32,        //number of hidden nodes
                    0.001,     //l2 regularization
                    new RmsProp(0.0005), null, false
            );

    public static ActorCriticFactoryCompGraphStdDense.Configuration NET_CONF_CG =
            new ActorCriticFactoryCompGraphStdDense.Configuration(
                    2,         //number of layers
                    128,       //number of hidden nodes
                    0.00001,   //l2 regularization
                    new RmsProp(0.005), null, true
            );

    @Test
    public void testModelLoadSave() throws IOException {
        ActorCriticSeparate acs = new ActorCriticFactorySeparateStdDense(NET_CONF).buildActorCritic(new int[] {7}, 5);

        File fileValue = File.createTempFile("rl4j-value-", ".model");
        File filePolicy = File.createTempFile("rl4j-policy-", ".model");
        acs.save(fileValue.getAbsolutePath(), filePolicy.getAbsolutePath());

        ActorCriticSeparate acs2 = ActorCriticSeparate.load(fileValue.getAbsolutePath(), filePolicy.getAbsolutePath());

        assertEquals(acs.valueNet, acs2.valueNet);
        assertEquals(acs.policyNet, acs2.policyNet);

        ActorCriticCompGraph accg = new ActorCriticFactoryCompGraphStdDense(NET_CONF_CG).buildActorCritic(new int[] {37}, 43);

        File file = File.createTempFile("rl4j-cg-", ".model");
        accg.save(file.getAbsolutePath());

        ActorCriticCompGraph accg2 = ActorCriticCompGraph.load(file.getAbsolutePath());

        assertEquals(accg.cg, accg2.cg);
    }

    @Test
    public void testLoss() {
        ActivationSoftmax activation = new ActivationSoftmax();
        ActorCriticLoss loss = new ActorCriticLoss();
        double n = 10;
        double eps = 1e-5;
        double maxRelError = 1e-3;

        for (double i = eps; i < n; i++) {
            for (double j = eps; j < n; j++) {
                INDArray labels = Nd4j.create(new double[] {i / n, 1 - i / n});
                INDArray output = Nd4j.create(new double[] {j / n, 1 - j / n});
                INDArray gradient = loss.computeGradient(labels, output, activation, null);

                output = Nd4j.create(new double[] {j / n, 1 - j / n});
                double score = loss.computeScore(labels, output, activation, null, false);
                INDArray output1 = Nd4j.create(new double[] {j / n + eps, 1 - j / n});
                double score1 = loss.computeScore(labels, output1, activation, null, false);
                INDArray output2 = Nd4j.create(new double[] {j / n, 1 - j / n + eps});
                double score2 = loss.computeScore(labels, output2, activation, null, false);

                double gradient1 = (score1 - score) / eps;
                double gradient2 = (score2 - score) / eps;
                double error1 = gradient1 - gradient.getDouble(0);
                double error2 = gradient2 - gradient.getDouble(1);
                double relError1 = error1 / gradient.getDouble(0);
                double relError2 = error2 / gradient.getDouble(1);
                System.out.println(gradient.getDouble(0) + "  " + gradient1 + " " + relError1);
                System.out.println(gradient.getDouble(1) + "  " + gradient2 + " " + relError2);
                assertTrue(gradient.getDouble(0) < maxRelError || Math.abs(relError1) < maxRelError);
                assertTrue(gradient.getDouble(1) < maxRelError || Math.abs(relError2) < maxRelError);
            }
        }
    }
}
