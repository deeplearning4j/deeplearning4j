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

package org.nd4j.autodiff.listeners.records;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LossCurve {
    @Getter
    private List<String> lossNames;
    @Getter
    private INDArray lossValues;

    public LossCurve(List<Loss> losses){
        lossNames = Collections.unmodifiableList(losses.get(0).getLossNames());
        int numLossValues = losses.get(0).lossValues().length;
        lossValues = Nd4j.create(DataType.FLOAT, losses.size(), losses.get(0).lossValues().length);

        for(int i = 0 ; i < losses.size() ; i++){
            Loss l = losses.get(i);
            Preconditions.checkArgument(l.getLossNames().equals(lossNames),
                    "Loss names for loss %s differ from others.  Expected %s, got %s",
                    i, lossNames, l.getLossNames());

            Preconditions.checkArgument(l.getLosses().length == numLossValues,
                    "Number of loss values for loss %s differ from others.  Expected %s, got %s",
                    i, numLossValues, l.getLosses().length);

            lossValues = lossValues.putRow(i, Nd4j.createFromArray(l.getLosses()).castTo(DataType.FLOAT));
        }
    }

    public LossCurve(double[] lossValues, List<String> lossNames){
        this.lossValues = Nd4j.createFromArray(new double[][]{ lossValues}).castTo(DataType.FLOAT);
        this.lossNames = lossNames;
    }

    protected LossCurve(INDArray lossValues, List<String> lossNames){
        Preconditions.checkArgument(lossValues.rank() == 2, "lossValues must have a rank of 2, got %s", lossValues.rank());
        Preconditions.checkArgument(lossValues.dataType() == DataType.FLOAT, "lossValues must be type FLOAT, got %s", lossValues.dataType());
        this.lossValues = lossValues;
        this.lossNames = lossNames;
    }

    public List<Loss> losses(){
        List<Loss> losses = new ArrayList<>();
        for(int i = 0 ; i < lossValues.size(0) ; i++){
            losses.add(new Loss(lossNames, lossValues.getRow(i).toDoubleVector()));
        }
        return losses;
    }

    /**
     * Get the mean loss for a given epoch
     *
     * If epoch is negative, counts backwards from the end.
     * E.g. losses(-1) gets the last epoch.
     *
     * @param epoch The epoch to get.  If negative, returns results for the epoch that many epochs from the end
     */
    public Loss meanLoss(int epoch){
        if(epoch >= 0){
            return new Loss(lossNames, lossValues.getRow(epoch).toDoubleVector());
        } else {
            return new Loss(lossNames, lossValues.getRow(lossValues.rows() + epoch).toDoubleVector());
        }
    }

    /**
     * Get the mean loss for the last epoch.
     */
    public Loss lastMeanLoss(){
        return meanLoss(-1);
    }

    /**
     * Return all mean loss values for a given variable
     */
    public float[] meanLoss(@NonNull String lossName){

        int idx = lossNames.indexOf(lossName);

        Preconditions.checkArgument(idx >= 0, "No loss value for %s.  Existing losses: %s", lossName, lossNames);

        float[] loss = new float[(int) lossValues.size(0)];
        for(int i = 0 ; i < lossValues.size(0) ; i++){
            loss[i] = lossValues.getFloat(i, idx);
        }
        return loss;
    }

    /**
     * Return all mean loss values for a given variable
     */
    public float[] meanLoss(@NonNull SDVariable loss){
        return meanLoss(loss.name());
    }

    /**
     * Return the mean loss value for a given variable on a given epoch.
     *
     * See {@link #meanLoss(int)}
     */
    public float meanLoss(@NonNull String lossName, int epoch){

        int idx = lossNames.indexOf(lossName);

        Preconditions.checkArgument(idx >= 0, "No loss value for %s.  Existing losses: %s", lossName, lossNames);

        if(epoch >= 0) {
            return lossValues.getFloat(epoch, idx);
        } else {
            return lossValues.getFloat(lossValues.rows() + epoch, idx);
        }
    }

    /**
     * Return the mean loss value for a given variable on a given epoch.
     *
     * See {@link #meanLoss(int)}
     */
    public float meanLoss(@NonNull SDVariable loss, int epoch){
        return meanLoss(loss.name(), epoch);
    }

    /**
     * Return the mean loss value for a given variable on the last epoch.
     */
    public float lastMeanLoss(@NonNull String lossName){

        int idx = lossNames.indexOf(lossName);

        Preconditions.checkArgument(idx >= 0, "No loss value for %s.  Existing losses: %s", lossName, lossNames);

        return lossValues.getFloat(lossValues.rows() - 1, idx);
    }

    /**
     * Return the mean loss value for a given variable on the last epoch.
     */
    public float lastMeanLoss(@NonNull SDVariable loss){
        return lastMeanLoss(loss.name());
    }

    /**
     * Return the loss delta between the last epoch and the one before it.
     * Equivalent to meanLoss(-1) - meanLoss(-2).
     * A positive delta means the loss is increasing, and a negative delta means it is decreasing.
     */
    public Loss lastMeanDelta(){
        return lastMeanLoss().sub(meanLoss(-2));
    }

    /**
     * Return the loss delta between the last epoch and the one before it, for a given variable.
     * Equivalent to meanLoss(-1) - meanLoss(-2).
     * A positive delta means the loss is increasing, and a negative delta means it is decreasing.
     */
    public double lastMeanDelta(String lossName){
        return lastMeanDelta().getLoss(lossName);
    }

    /**
     * Return the loss delta between the last epoch and the one before it, for a given variable.
     * Equivalent to meanLoss(-1) - meanLoss(-2).
     * A positive delta means the loss is increasing, and a negative delta means it is decreasing.
     */
    public double lastMeanDelta(SDVariable loss){
        return lastMeanDelta(loss.name());
    }

    /**
     * Return a new LossCurve with the given losses added on as the most recent epoch
     */
    public LossCurve addLossAndCopy(Loss loss){
        return addLossAndCopy(loss.getLosses(), loss.lossNames());
    }

    /**
     * Return a new LossCurve with the given losses added on as the most recent epoch
     */
    public LossCurve addLossAndCopy(double[] values, List<String> lossNames){
        return new LossCurve(
                Nd4j.concat(0, lossValues,
                        Nd4j.createFromArray(new double[][]{values}).castTo(DataType.FLOAT)),
                lossNames);
    }
}
