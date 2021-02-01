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

package org.nd4j.linalg.learning.regularization;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.Axpy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * L2 regularization: very similar to {@link WeightDecay}, but is applied before the updater is applied, not after.
 * <br>
 * <br>
 * Implements updating as follows:<br>
 * {@code L = loss + l2 * 0.5 * sum_i w[i]^2}<br>
 * {@code w[i] -= updater(gradient[i] + l2 * w[i])<br>
 * That is, L2 regularization is applied before the updater (Adam/Nesterov/etc) is applied to the gradients. This differs
 * from {@link WeightDecay} mainly in that WeightDecay is applied after the updater.
 *
 * See also: {@link WeightDecay} which should generally be preferred in practice.<br>
 * See <a href="https://www.fast.ai/2018/07/02/adam-weight-decay/">https://www.fast.ai/2018/07/02/adam-weight-decay/</a>
 * for further details
 *
 * @author Alex Black
 */
@Data
public class L2Regularization implements Regularization {

    protected final ISchedule l2;

    /**
     * @param l2   L2 regularization coefficient
     */
    public L2Regularization(double l2) {
        this(new FixedSchedule(l2));
    }

    /**
     * @param l2 L2 regularization coefficient (schedule)
     */
    public L2Regularization(@JsonProperty("l2") @NonNull ISchedule l2) {
        this.l2 = l2;
    }

    @Override
    public ApplyStep applyStep(){
        return ApplyStep.BEFORE_UPDATER;
    }

    @Override
    public void apply(INDArray param, INDArray gradView, double lr, int iteration, int epoch) {
        //L = loss + l2 * 0.5 * sum_i x[i]^2
        //dL/dx[i] = dloss/dx[i] + l2 * x[i]
        double coeff = l2.valueAt(iteration, epoch);
        Nd4j.exec(new Axpy(param, gradView, gradView, coeff));    //Gradient = scale * param + gradient
    }

    @Override
    public double score(INDArray param, int iteration, int epoch) {
        //Score: L = 0.5 * sum_i x[i]^2
        double norm2 = param.norm2Number().doubleValue();   //Norm2 is sqrt(sum_i x[i]^2)
        return l2.valueAt(iteration, epoch) * 0.5 * norm2 * norm2;
    }

    @Override
    public Regularization clone() {
        return new L2Regularization(l2.clone());
    }
}
