package org.nd4j.autodiff.listeners;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.base.Preconditions;

import java.util.List;

@Data
public class Loss {

    private final List<String> lossNames;
    private final double[] losses;

    public Loss(@NonNull List<String> lossNames, @NonNull double[] losses){
        Preconditions.checkState(lossNames.size() == losses.length, "Expected equal number of loss names and loss values");
        this.lossNames = lossNames;
        this.losses = losses;
    }

    public int numLosses(){
        return lossNames.size();
    }

    public List<String> lossNames(){
        return lossNames;
    }

    public double[] lossValues(){
        return losses;
    }

    public double getLoss(@NonNull String lossName){
        int idx = lossNames.indexOf(lossName);
        Preconditions.checkState(idx >= 0, "No loss with name \"%s\" exists. All loss names: %s", lossName, lossNames);
        return losses[idx];
    }

    public double totalLoss(){
        double sum = 0.0;
        for(double d : losses){
            sum += d;
        }
        return sum;
    }
}
