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

package org.nd4j.autodiff.listeners;

import java.util.ArrayList;
import java.util.Collections;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.common.base.Preconditions;

import java.util.List;

/**
 * Loss class - represents the loss (score) for the network, for one iteration. Provides a breakdown of all the loss components
 *
 * @author Alex Black
 */
@Data
public class Loss {

    private final List<String> lossNames;
    private final double[] losses;

    /**
     * @param lossNames Names of the losses
     * @param losses    Values for each loss. Must be same length as lossNames
     */
    public Loss(@NonNull List<String> lossNames, @NonNull double[] losses) {
        Preconditions.checkState(lossNames.size() == losses.length, "Expected equal number of loss names and loss values");
        this.lossNames = lossNames;
        this.losses = losses;
    }

    /**
     * @return Number of loss values (i.e., length of lossNames and losses)
     */
    public int numLosses() {
        return lossNames.size();
    }

    /**
     * @return Names of all of the loss components
     */
    public List<String> lossNames() {
        return lossNames;
    }

    /**
     * @return Values corresponding to each of the losses (same order as lossNames())
     */
    public double[] lossValues() {
        return losses;
    }

    /**
     * Get the specified loss by name
     *
     * @param lossName Name of the loss (must exist)
     * @return Specified loss value
     */
    public double getLoss(@NonNull String lossName) {
        int idx = lossNames.indexOf(lossName);
        Preconditions.checkState(idx >= 0, "No loss with name \"%s\" exists. All loss names: %s", lossName, lossNames);
        return losses[idx];
    }

    /**
     * @return The total loss (sum of all loss components)
     */
    public double totalLoss() {
        double sum = 0.0;
        for (double d : losses) {
            sum += d;
        }
        return sum;
    }

    public Loss copy() {
        return new Loss(lossNames, losses);
    }

    public static Loss sum(List<Loss> losses) {

        if (losses.isEmpty())
            return new Loss(Collections.<String>emptyList(), new double[0]);

        double[] lossValues = new double[losses.get(0).losses.length];
        List<String> lossNames = new ArrayList<>(losses.get(0).lossNames);

        for (int i = 0; i < losses.size(); i++) {
            Loss l = losses.get(i);
            Preconditions.checkState(l.losses.length == lossValues.length,
                    "Loss %s has %s losses, the others before it had %s.", i, l.losses.length, lossValues.length);

            Preconditions.checkState(l.lossNames.equals(lossNames),
                    "Loss %s has different loss names from the others before it.  Expected %s, got %s.",
                    i, lossNames, l.lossNames);

            for (int j = 0; j < lossValues.length; j++)
                lossValues[j] += l.losses[j];

        }

        return new Loss(lossNames, lossValues);
    }

    public static Loss average(List<Loss> losses) {
        Loss sum = sum(losses);

        for (int i = 0; i < sum.losses.length; i++) {
            sum.losses[i] /= losses.size();
        }

        return sum;
    }

    public static Loss add(Loss a, Loss b) {
        Preconditions.checkState(a.lossNames.equals(b.lossNames),
                "Loss names differ.  First loss has names %s, second has names %s.",
                a.lossNames, b.lossNames);

        double[] lossValues = new double[a.losses.length];
        for (int i = 0; i < lossValues.length; i++)
            lossValues[i] = a.losses[i] + b.losses[i];

        return new Loss(a.lossNames, lossValues);
    }

    public static Loss sub(Loss a, Loss b) {
        Preconditions.checkState(a.lossNames.equals(b.lossNames),
                "Loss names differ.  First loss has names %s, second has names %s.",
                a.lossNames, b.lossNames);

        double[] lossValues = new double[a.losses.length];
        for (int i = 0; i < lossValues.length; i++)
            lossValues[i] = a.losses[i] - b.losses[i];

        return new Loss(a.lossNames, lossValues);
    }

    public static Loss div(Loss a, Number b) {
        double[] lossValues = new double[a.losses.length];
        for (int i = 0; i < lossValues.length; i++)
            lossValues[i] = a.losses[i] / b.doubleValue();

        return new Loss(a.lossNames, lossValues);
    }

    public Loss add(Loss other) {
        return add(this, other);
    }

    public Loss sub(Loss other) {
        return sub(this, other);
    }

    public Loss plus(Loss other) {
        return add(this, other);
    }

    public Loss minus(Loss other) {
        return sub(this, other);
    }

    public Loss div(Number other) {
        return div(this, other);
    }

}
