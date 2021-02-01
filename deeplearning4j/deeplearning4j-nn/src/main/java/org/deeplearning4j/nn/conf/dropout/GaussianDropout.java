/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
@JsonIgnoreProperties({"noise"})
@EqualsAndHashCode(exclude = {"noise"})
public class GaussianDropout implements IDropout {

    private final double rate;
    private final ISchedule rateSchedule;
    private transient INDArray noise;

    /**
     * @param rate Rate parameter, see {@link GaussianDropout}
     */
    public GaussianDropout(double rate){
        this(rate, null);
    }

    /**
     * @param rateSchedule Schedule for rate parameter, see {@link GaussianDropout}
     */
    public GaussianDropout(ISchedule rateSchedule){
        this(Double.NaN, rateSchedule);
    }

    protected GaussianDropout(@JsonProperty("rate") double rate, @JsonProperty("rateSchedule") ISchedule rateSchedule){
        this.rate = rate;
        this.rateSchedule = rateSchedule;
    }

    @Override
    public INDArray applyDropout(INDArray inputActivations, INDArray output, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr) {
        double r;
        if(rateSchedule != null){
            r = rateSchedule.valueAt(iteration, epoch);
        } else {
            r = rate;
        }

        double stdev = Math.sqrt(r / (1.0 - r));

        noise = workspaceMgr.createUninitialized(ArrayType.INPUT, output.dataType(), inputActivations.shape(), inputActivations.ordering());
        Nd4j.getExecutioner().exec(new GaussianDistribution(noise, 1.0, stdev));

        return Nd4j.getExecutioner().exec(new MulOp(inputActivations, noise, output))[0];
    }

    @Override
    public INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch) {
        Preconditions.checkState(noise != null, "Cannot perform backprop: GaussianDropout noise array is absent (already cleared?)");
        //out = in*y, where y ~ N(1, stdev)
        //dL/dIn = dL/dOut * dOut/dIn = y * dL/dOut
        Nd4j.getExecutioner().exec(new MulOp(gradAtOutput, noise, gradAtInput));
        noise = null;
        return gradAtInput;
    }

    @Override
    public void clear() {
        noise = null;
    }

    @Override
    public GaussianDropout clone() {
        return new GaussianDropout(rate, rateSchedule == null ? null : rateSchedule.clone());
    }
}
