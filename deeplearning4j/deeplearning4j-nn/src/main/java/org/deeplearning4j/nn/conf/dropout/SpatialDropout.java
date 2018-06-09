package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.val;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Spatial dropout: can only be applied to 4D (convolutional) activations.
 * Dropout mask is generated along the depth dimension, and is applied to each x/y location in an image.<br>
 * Note that the dropout mask is generated independently for each example: i.e., a dropout mask of shape [minibatch, channels]
 * is generated and applied to activations of shape [minibatch, channels, height, width]
 * <p>
 * Reference: Efficient Object Localization Using Convolutional Networks: https://arxiv.org/abs/1411.4280
 *
 * @author Alex Black
 */
@Data
@JsonIgnoreProperties({"mask"})
@EqualsAndHashCode(exclude = {"mask"})
public class SpatialDropout implements IDropout {

    private double p;
    private ISchedule pSchedule;
    private transient INDArray mask;

    /**
     * @param activationRetainProbability Probability of retaining an activation - see {@link Dropout} javadoc
     */
    public SpatialDropout(double activationRetainProbability) {
        this(activationRetainProbability, null);
        if (activationRetainProbability < 0.0) {
            throw new IllegalArgumentException("Activation retain probability must be > 0. Got: " + activationRetainProbability);
        }
        if (activationRetainProbability == 0.0) {
            throw new IllegalArgumentException("Invalid probability value: Dropout with 0.0 probability of retaining "
                    + "activations is not supported");
        }
    }

    /**
     * @param activationRetainProbabilitySchedule Schedule for probability of retaining an activation - see {@link Dropout} javadoc
     */
    public SpatialDropout(ISchedule activationRetainProbabilitySchedule) {
        this(Double.NaN, activationRetainProbabilitySchedule);
    }

    protected SpatialDropout(@JsonProperty("p") double activationRetainProbability,
                             @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule) {
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
    }


    @Override
    public INDArray applyDropout(INDArray inputActivations, INDArray output, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr) {
        Preconditions.checkArgument(inputActivations.rank() == 5 || inputActivations.rank() == 4
                || inputActivations.rank() == 3, "Cannot apply spatial dropout to activations of rank %s: " +
                "spatial dropout can only be used for rank 3, 4 or 5 activations (input activations shape: %s)"
                , inputActivations.rank(), inputActivations.shape());

        double currP;
        if (pSchedule != null) {
            currP = pSchedule.valueAt(iteration, epoch);
        } else {
            currP = p;
        }

        val minibatch = inputActivations.size(0);
        val dim1 = inputActivations.size(1);
        mask = workspaceMgr.createUninitialized(ArrayType.INPUT, minibatch, dim1).assign(1.0);
        Nd4j.getExecutioner().exec(new DropOutInverted(mask, currP));

        Broadcast.mul(inputActivations, mask, output, 0, 1);
        return output;
    }

    @Override
    public INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch) {
        Preconditions.checkState(mask != null, "Cannot perform backprop: Dropout mask array is absent (already cleared?)");
        //Mask has values 0 or 1/p
        //dL/dIn = dL/dOut * dOut/dIn = dL/dOut * (0 if dropped, or 1/p otherwise)
        Broadcast.mul(gradAtOutput, mask, gradAtInput, 0, 1);
        mask = null;
        return gradAtInput;
    }

    @Override
    public void clear() {
        mask = null;
    }

    @Override
    public IDropout clone() {
        return new SpatialDropout(p, pSchedule);
    }
}
