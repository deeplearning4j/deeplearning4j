package org.deeplearning4j.nn.conf.dropout;

import lombok.Data;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Implements standard (inverted) dropout.<br>
 * <br>
 * Regarding dropout probability. This is the probability of <it>retaining</it> each input activation value for a layer.
 * Thus, each input activation x is independently set to:<br>
 * x <- 0, with probability 1-p<br>
 * x <- x/p with probability p<br>
 * Note that this "inverted" dropout scheme maintains the expected value of activations - i.e., E(x) is the same before
 * and after dropout.<br>
 * Dropout schedules (i.e., varying probability p as a function of iteration/epoch) are also supported.<br>
 * <br>
 * Other libraries (notably, Keras) use p == probability(<i>dropping</i> an activation)<br>
 * In DL4J, {@code new Dropout(x)} will keep an input activation with probability x, and set to 0 with probability 1-x.<br>
 * Thus, a dropout value of 1.0 is functionally equivalent to no dropout: i.e., 100% probability of retaining
 * each input activation.<br>
 * <p>
 * Note 1: As per all IDropout instances, dropout is applied at training time only - and is automatically not applied at
 * test time (for evaluation, etc)<br>
 * Note 2: Care should be taken when setting lower (probability of retaining) values for (too much information may be
 * lost with aggressive (very low) dropout values).<br>
 * Note 3: Frequently, dropout is not applied to (or, has higher retain probability for) input (first layer)
 * layers. Dropout is also often not applied to output layers.<br>
 * Note 4: Implementation detail (most users can ignore): DL4J uses inverted dropout, as described here:
 * <a href="http://cs231n.github.io/neural-networks-2/">http://cs231n.github.io/neural-networks-2/</a>
 * </p>
 * <br>
 * See: Srivastava et al. 2014: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
 * <a href="http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf">http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf</a>
 *
 * @author Alex Black
 */
@Data
public class Dropout implements IDropout {

    private double p;
    private ISchedule pSchedule;

    /**
     * @param activationRetainProbability Probability of retaining an activation - see {@link Dropout} javadoc
     */
    public Dropout(double activationRetainProbability) {
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
    public Dropout(ISchedule activationRetainProbabilitySchedule){
        this(Double.NaN, activationRetainProbabilitySchedule);
    }

    protected Dropout(@JsonProperty("p") double activationRetainProbability, @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule) {
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
    }


    @Override
    public INDArray applyDropout(INDArray inputActivations, INDArray output, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr) {
        double currP;
        if(pSchedule != null){
            currP = pSchedule.valueAt(iteration, epoch);
        } else {
            currP = p;
        }

        Nd4j.getExecutioner().exec(new DropOutInverted(inputActivations, output, currP));
        return output;
    }

    @Override
    public INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch) {
        throw new RuntimeException("Not yet implemented");
    }

    @Override
    public void clear() {
        //TODO
    }

    @Override
    public Dropout clone() {
        return new Dropout(p, pSchedule == null ? null : pSchedule.clone());
    }
}
