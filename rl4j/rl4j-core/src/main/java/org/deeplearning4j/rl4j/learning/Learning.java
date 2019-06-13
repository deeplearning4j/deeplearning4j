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

package org.deeplearning4j.rl4j.learning;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.ILearningInitializer;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/27/16.
 *
 * Useful factorisations and helper methods for class inheriting
 * ILearning.
 *
 * Big majority of training method should inherit this
 *
 */
@Slf4j
public abstract class Learning<O extends Encodable, A, AS extends ActionSpace<A>, NN extends NeuralNet>
                implements ILearning<O, A, AS>, NeuralNetFetchable<NN> {
    @Getter
    final private Random random;
    @Getter @Setter
    private int stepCounter = 0;
    @Getter @Setter
    private int epochCounter = 0;
    @Getter @Setter
    private IHistoryProcessor historyProcessor = null;

    protected ILearningInitializer<O, A, AS> initializer  = new LearningInitializer<O, A, AS>();

    public Learning(LConfiguration conf) {
        random = new Random(conf.getSeed());
    }

    public static Integer getMaxAction(INDArray vector) {
        return Nd4j.argMax(vector, Integer.MAX_VALUE).getInt(0);
    }

    public static <O extends Encodable, A, AS extends ActionSpace<A>> INDArray getInput(MDP<O, A, AS> mdp, O obs) {
        INDArray arr = Nd4j.create(obs.toArray());
        int[] shape = mdp.getObservationSpace().getShape();
        if (shape.length == 1)
            return arr.reshape(new long[] {1, arr.length()});
        else
            return arr.reshape(shape);
    }

    public static int[] makeShape(int size, int[] shape) {
        int[] nshape = new int[shape.length + 1];
        nshape[0] = size;
        System.arraycopy(shape, 0, nshape, 1, shape.length);
        return nshape;
    }

    public static int[] makeShape(int batch, int[] shape, int length) {
        int[] nshape = new int[3];
        nshape[0] = batch;
        nshape[1] = 1;
        for (int i = 0; i < shape.length; i++) {
            nshape[1] *= shape[i];
        }
        nshape[2] = length;
        return nshape;
    }

    protected abstract DataManager getDataManager();

    public abstract NN getNeuralNet();

    public int incrementStep() {
        return stepCounter++;
    }

    public int incrementEpoch() {
        return epochCounter++;
    }

    public void setHistoryProcessor(HistoryProcessor.Configuration conf) {
        setHistoryProcessor(new HistoryProcessor(conf));
    }

    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;

        if(historyProcessor == null) {
            initializer = new LearningInitializer<O, A, AS>();
        } else {
            initializer = new HistoryProcessorLearningInitializer<O, A, AS>(historyProcessor);
        }
    }

    public INDArray getInput(O obs) {
        return getInput(getMdp(), obs);
    }

    public InitMdp<O> initMdp() {
        getNeuralNet().reset();
        return initializer.initMdp(getMdp());
    }

    @AllArgsConstructor
    @Value
    public static class InitMdp<O> {
        int steps;
        O lastObs;
        double reward;
    }

}
