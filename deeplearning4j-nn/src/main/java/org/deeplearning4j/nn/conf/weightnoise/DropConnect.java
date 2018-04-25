package org.deeplearning4j.nn.conf.weightnoise;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * DropConnect, based on Wan et. al 2013 - "Regularization of Neural Networks using DropConnect"<br>
 * Sets weights randomly to 0 with some probability, or leaves them unchanged.
 *
 * @author Alex Black
 */
@Data
public class DropConnect implements IWeightNoise {

    private double weightRetainProb;
    private ISchedule weightRetainProbSchedule;
    private boolean applyToBiases;

    /**
     * @param weightRetainProbability Probability of retaining a weight
     */
    public DropConnect(double weightRetainProbability) {
        this(weightRetainProbability, false);
    }

    /**
     * @param weightRetainProbability Probability of retaining a weight
     * @param applyToBiases If true: apply to biases (default: weights only)
     */
    public DropConnect(double weightRetainProbability, boolean applyToBiases) {
        this(weightRetainProbability, null, applyToBiases);
    }

    /**
     * @param weightRetainProbSchedule Probability (schedule) of retaining a weight
     */
    public DropConnect(ISchedule weightRetainProbSchedule){
        this(Double.NaN, weightRetainProbSchedule, false);
    }

    /**
     * @param weightRetainProbSchedule Probability (schedule) of retaining a weight
     * @param applyToBiases If true: apply to biases (default: weights only)
     */
    public DropConnect(ISchedule weightRetainProbSchedule, boolean applyToBiases){
        this(Double.NaN, weightRetainProbSchedule, applyToBiases);
    }

    private DropConnect(@JsonProperty("weightRetainProbability") double weightRetainProbability,
                        @JsonProperty("weightRetainProbSchedule") ISchedule weightRetainProbSchedule,
                        @JsonProperty("applyToBiases") boolean applyToBiases) {
        this.weightRetainProb = weightRetainProbability;
        this.weightRetainProbSchedule = weightRetainProbSchedule;
        this.applyToBiases = applyToBiases;
    }

    @Override
    public INDArray getParameter(Layer layer, String paramKey, int iteration, int epoch, boolean train, LayerWorkspaceMgr workspaceMgr) {
        ParamInitializer init = layer.conf().getLayer().initializer();
        INDArray param = layer.getParam(paramKey);

        double p;
        if(weightRetainProbSchedule == null){
            p = weightRetainProb;
        } else {
            p = weightRetainProbSchedule.valueAt(iteration, epoch);
        }

        if (train && init.isWeightParam(layer.conf().getLayer(), paramKey)
                || (applyToBiases && init.isBiasParam(layer.conf().getLayer(), paramKey))) {
            INDArray out = workspaceMgr.createUninitialized(ArrayType.INPUT, param.shape(), param.ordering());
            Nd4j.getExecutioner().exec(new DropOut(param, out, p));
            return out;
        }
        return param;
    }

    @Override
    public DropConnect clone() {
        return new DropConnect(weightRetainProb, weightRetainProbSchedule, applyToBiases);
    }
}
