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
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * L1 regularization: Implements updating as follows:<br>
 * {@code L = loss + l1 * sum_i abs(w[i])}<br>
 * {@code w[i] -= updater(gradient[i] + l1 * sign(w[i])) - where sign(w[i]) is +/- 1<br>
 * Note that L1 regularization is applied before the updater (Adam/Nesterov/etc) is applied.
 *
 * @author Alex Black
 */
@Data
public class L1Regularization implements Regularization {

    protected final ISchedule l1;

    /**
     * @param l1   l1 regularization coefficient
     */
    public L1Regularization(double l1) {
        this(new FixedSchedule(l1));
    }

    /**
     * @param l1 L1 regularization coefficient (schedule)
     */
    public L1Regularization(@JsonProperty("l1") @NonNull ISchedule l1) {
        this.l1 = l1;
    }

    @Override
    public ApplyStep applyStep(){
        return ApplyStep.BEFORE_UPDATER;
    }

    @Override
    public void apply(INDArray param, INDArray gradView, double lr, int iteration, int epoch) {
        //L = loss + l1 * sum_i abs(x[i])
        //dL/dx[i] = dloss/dx[i] + l1 * sign(x[i])
        //where sign(x[i]) is -1 or 1
        double coeff = l1.valueAt(iteration, epoch);
        INDArray sign = Transforms.sign(param, true);
        Nd4j.exec(new Axpy(sign, gradView, gradView, coeff));    //Gradient = l1 * sign(param) + gradient
    }

    @Override
    public double score(INDArray param, int iteration, int epoch) {
        return l1.valueAt(iteration, epoch) * param.norm1Number().doubleValue();
    }

    @Override
    public Regularization clone() {
        return new L1Regularization(l1.clone());
    }
}
