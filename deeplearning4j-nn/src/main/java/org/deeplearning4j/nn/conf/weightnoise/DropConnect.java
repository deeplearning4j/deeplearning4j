package org.deeplearning4j.nn.conf.weightnoise;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;


@Data
public class DropConnect implements IWeightNoise {

    private double weightRetainProb;
    private ISchedule weightRetainProbSchedule;
    private boolean applyToBiases;

    public DropConnect(double weightRetainProbability) {
        this(weightRetainProbability, false);
    }

    public DropConnect(double weightRetainProbability, boolean applyToBiases) {
        this(weightRetainProbability, null, applyToBiases);
    }

    public DropConnect(ISchedule weightRetainProbSchedule){
        this(Double.NaN, weightRetainProbSchedule, false);
    }

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
    public INDArray getParameter(Layer layer, String paramKey, int iteration, int epoch, boolean train) {
        ParamInitializer init = layer.conf().getLayer().initializer();
        INDArray param = layer.getParam(paramKey);

        double p;
        if(weightRetainProbSchedule == null){
            p = weightRetainProb;
        } else {
            p = weightRetainProbSchedule.valueAt(iteration, epoch);
        }

        if (train && init.isWeightParam(paramKey) || (applyToBiases && init.isBiasParam(paramKey))) {
            INDArray out = Nd4j.createUninitialized(param.shape(), param.ordering());
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
