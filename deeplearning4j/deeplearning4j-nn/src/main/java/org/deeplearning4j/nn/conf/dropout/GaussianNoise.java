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

package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldAddOp;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Applies additive, mean-zero Gaussian noise to the input - i.e., x = x + N(0,stddev).<br>
 * Note that this differs from {@link GaussianDropout}, which applies <it>multiplicative</it> mean-1 N(1,s) noise.<br>
 * Note also that schedules for the standard deviation value can also be used.
 *
 * @author Alex Black
 */
@Data
public class GaussianNoise implements IDropout {

    private double stddev;
    private ISchedule stddevSchedule;

    /**
     * @param stddev Standard deviation for the mean 0 Gaussian noise
     */
    public GaussianNoise(double stddev){
        this(stddev, null);
    }

    /**
     * @param stddevSchedule Schedule for standard deviation for the mean 0 Gaussian noise
     */
    public GaussianNoise(ISchedule stddevSchedule){
        this(Double.NaN, stddevSchedule);
    }

    protected GaussianNoise(@JsonProperty("stddev") double stddev, @JsonProperty("stddevSchedule") ISchedule stddevSchedule){
        this.stddev = stddev;
        this.stddevSchedule = stddevSchedule;
    }

    @Override
    public INDArray applyDropout(INDArray inputActivations, INDArray output, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr) {
        double currS;
        if(stddevSchedule != null){
            currS = stddevSchedule.valueAt(iteration, epoch);
        } else {
            currS = stddev;
        }

        INDArray noise = Nd4j.createUninitialized(inputActivations.shape(), inputActivations.ordering());
        Nd4j.getExecutioner().exec(new GaussianDistribution(noise, 0, currS));

        Nd4j.getExecutioner().exec(new OldAddOp(inputActivations, noise, output));
        return output;
    }

    @Override
    public INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch) {
        //dL/dIn = dL/dOut * dOut/dIn, with dOut/dIn = 1
        if(gradAtInput == gradAtOutput){
            //Same array (in-place result)
            return gradAtInput;
        } else {
            return gradAtInput.assign(gradAtOutput);
        }
    }

    @Override
    public void clear() {
        //No op
    }

    @Override
    public IDropout clone() {
        return new GaussianNoise(stddev, stddevSchedule);
    }
}
