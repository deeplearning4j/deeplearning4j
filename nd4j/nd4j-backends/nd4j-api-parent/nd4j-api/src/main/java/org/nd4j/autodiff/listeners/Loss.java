package org.nd4j.autodiff.listeners;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.base.Preconditions;

import java.util.List;

/**
 * Loss class - represents the loss (score) for the network. Provides a breakdown of all the loss components
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
}
