/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.agent.learning.algorithm.actorcritic;

import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * A helper class for the Actor Critic update algorithm. The algorithm is the same whether it's used with a RNN or
 * not but, the shape of INDArrays are different. This class handles the recurrent case.
 */
public class RecurrentActorCriticHelper extends ActorCriticHelper {
    private final int actionSpaceSize;

    public RecurrentActorCriticHelper(int actionSpaceSize) {
        this.actionSpaceSize = actionSpaceSize;
    }

    @Override
    protected INDArray createFeatureArray(int size, long[] observationShape) {
        return INDArrayHelper.createRnnBatchForShape(size, observationShape);
    }

    @Override
    public INDArray createValueLabels(int trainingBatchSize) {
        return Nd4j.create(1, 1, trainingBatchSize);
    }

    @Override
    public INDArray createPolicyLabels(int trainingBatchSize) {
        return Nd4j.zeros(1, actionSpaceSize, trainingBatchSize);
    }

    @Override
    protected void setFeature(INDArray features, long idx, INDArray data) {
        getElementAtIndex(features, idx).assign(data);
    }

    @Override
    public void setPolicy(INDArray policy, long idx, int action, double advantage) {
        policy.putScalar(0, action, idx, advantage);
    }

    private INDArray getElementAtIndex(INDArray array, long idx) {
        return array.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(idx));
    }
}
