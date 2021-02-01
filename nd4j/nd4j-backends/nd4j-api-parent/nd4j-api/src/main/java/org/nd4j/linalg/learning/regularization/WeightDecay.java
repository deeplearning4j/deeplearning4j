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

package org.nd4j.linalg.learning.regularization;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.Axpy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
public class WeightDecay implements Regularization {

    protected final ISchedule coeff;
    protected final boolean applyLR;

    /**
     * @param coeff   Weight decay regularization coefficient
     * @param applyLR If true, multiply the regularization coefficient by the current learning rate. If false, do not multiply by LR.
     */
    public WeightDecay(double coeff, boolean applyLR) {
        this(new FixedSchedule(coeff), applyLR);
    }

    /**
     * @param coeff   Weight decay regularization coefficient (schedule)
     * @param applyLR If true, multiply the regularization coefficient by the current learning rate. If false, do not multiply by LR.
     */
    public WeightDecay(@JsonProperty("coeff") @NonNull ISchedule coeff, @JsonProperty("applyLR") boolean applyLR){
        this.coeff = coeff;
        this.applyLR = applyLR;
    }

    @Override
    public ApplyStep applyStep() {
        return ApplyStep.POST_UPDATER;
    }

    @Override
    public void apply(INDArray param, INDArray gradView, double lr, int iteration, int epoch) {
        //L = loss + coeff * 0.5 * sum_i x[i]^2
        //dL/dx[i] = coeff * x[i]
        //update(x[i]) = coeff * x[i] * ( applyLR ? lr : )
        double scale = coeff.valueAt(iteration, epoch);
        if(applyLR){
            scale *= lr;
        }
        Nd4j.exec(new Axpy(param, gradView, gradView, scale));    //update = scale * param + update
    }

    @Override
    public double score(INDArray param, int iteration, int epoch) {
        //Score: L = 0.5 * sum_i x[i]^2
        double norm2 = param.norm2Number().doubleValue();   //Norm2 is sqrt(sum_i x[i]^2)
        return coeff.valueAt(iteration, epoch) * 0.5 * norm2 * norm2;
    }

    @Override
    public Regularization clone() {
        return new WeightDecay(coeff.clone(), applyLR);
    }
}
