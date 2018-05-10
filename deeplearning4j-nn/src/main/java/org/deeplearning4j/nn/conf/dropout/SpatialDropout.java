package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Spatial dropout: can only be applied to 4D (convolutional) activations.
 * Dropout mask is generated along the depth dimension, and is applied to each x/y location in an image.<br>
 * Note that the dropout mask is generated independently for each example: i.e., a dropout mask of shape [minibatch, channels]
 * is generated and applied to activations of shape [minibatch, channels, height, width]
 *
 * Reference: Efficient Object Localization Using Convolutional Networks: https://arxiv.org/abs/1411.4280
 *
 * @author Alex Black
 */
@Data
public class SpatialDropout implements IDropout {

    private double p;
    private ISchedule pSchedule;

    /**
     * @param activationRetainProbability Probability of retaining an activation - see {@link Dropout} javadoc
     */
    public SpatialDropout(double activationRetainProbability) {
        this(activationRetainProbability, null);
        if(activationRetainProbability < 0.0){
            throw new IllegalArgumentException("Activation retain probability must be > 0. Got: " + activationRetainProbability);
        }
        if(activationRetainProbability == 0.0){
            throw new IllegalArgumentException("Invalid probability value: Dropout with 0.0 probability of retaining "
                    + "activations is not supported");
        }
    }

    /**
     * @param activationRetainProbabilitySchedule Schedule for probability of retaining an activation - see {@link Dropout} javadoc
     */
    public SpatialDropout(ISchedule activationRetainProbabilitySchedule){
        this(Double.NaN, activationRetainProbabilitySchedule);
    }

    protected SpatialDropout(@JsonProperty("p") double activationRetainProbability,
                             @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule) {
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
    }


    @Override
    public INDArray applyDropout(@NonNull INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
        Preconditions.checkArgument(inputActivations.rank() == 4, "Cannot apply spatial dropout to activations of rank %s:" +
                " spatial dropout can only be used for rank 4 activations (input activations shape: %s)", inputActivations.rank(),
                inputActivations.shape());

        int minibatch = inputActivations.size(0);
        int channels = inputActivations.size(1);
        INDArray mc = Nd4j.ones(minibatch, channels);

        double currP;
        if(pSchedule != null){
            currP = pSchedule.valueAt(iteration, epoch);
        } else {
            currP = p;
        }

        Nd4j.getExecutioner().exec(new DropOutInverted(mc, currP));

        INDArray result;
        if(inPlace){
            result = inputActivations;
        } else {
            result = Nd4j.createUninitialized(inputActivations.shape(), 'c');
        }

        Broadcast.mul(inputActivations, mc, result, 0, 1);
        return result;
    }

    @Override
    public IDropout clone() {
        return new SpatialDropout(p, pSchedule);
    }
}
