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

package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/10/16.
 *
 * Boltzmann exploration is a stochastic policy wrt to the
 * exponential Q-values as evaluated by the dqn model.
 */
public class BoltzmannQ<OBSERVATION extends Encodable> extends Policy<Integer> {

    final private IDQN dqn;
    final private Random rnd;

    public BoltzmannQ(IDQN dqn, Random random) {
        this.dqn = dqn;
        this.rnd = random;
    }

    public IDQN getNeuralNet() {
        return dqn;
    }

    @Override
    public Integer nextAction(Observation obs) {
        INDArray output = dqn.output(obs).get(CommonOutputNames.QValues);
        INDArray exp = exp(output);

        double sum = exp.sum(1).getDouble(0);
        double picked = rnd.nextDouble() * sum;
        for (int i = 0; i < exp.columns(); i++) {
            if (picked < exp.getDouble(i))
                return i;
        }
        return -1;
    }

    @Deprecated
    public Integer nextAction(INDArray input) {
        INDArray output = dqn.output(input).get(CommonOutputNames.QValues);
        INDArray exp = exp(output);

        double sum = exp.sum(1).getDouble(0);
        double picked = rnd.nextDouble() * sum;
        for (int i = 0; i < exp.columns(); i++) {
            if (picked < exp.getDouble(i))
                return i;
        }
        return -1;
    }


}
