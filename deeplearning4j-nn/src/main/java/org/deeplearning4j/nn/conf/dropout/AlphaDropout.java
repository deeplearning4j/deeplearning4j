package org.deeplearning4j.nn.conf.dropout;

import lombok.NonNull;
import org.deeplearning4j.nn.conf.schedule.ISchedule;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.AlphaDropOut;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@JsonIgnoreProperties({"lastPValue", "alphaPrime", "a", "b"})
public class AlphaDropout implements IDropout {

    public static final double DEFAULT_ALPHA =  1.6732632423543772;
    public static final double DEFAULT_LAMBDA = 1.0507009873554804;


    private final double p;
    private final ISchedule pSchedule;
    private final double alpha;
    private final double lambda;

    private double lastPValue;
    private double alphaPrime;
    private double a;
    private double b;

    public AlphaDropout(double activationRetainProbability){
        this(activationRetainProbability, null, DEFAULT_ALPHA, DEFAULT_LAMBDA);
    }

    public AlphaDropout(@NonNull ISchedule activationRetainProbabilitySchedule){
        this(Double.NaN, activationRetainProbabilitySchedule, DEFAULT_ALPHA, DEFAULT_LAMBDA);
    }

    protected AlphaDropout(@JsonProperty("p")double activationRetainProbability,
                           @JsonProperty("pSchedule") ISchedule activationRetainProbabilitySchedule,
                           @JsonProperty("alpha") double alpha, @JsonProperty("lambda") double lambda ){
        this.p = activationRetainProbability;
        this.pSchedule = activationRetainProbabilitySchedule;
        this.alpha = alpha;
        this.lambda = lambda;

        this.alphaPrime = -lambda * alpha;
        if(activationRetainProbabilitySchedule == null){
            this.lastPValue = p;
            this.a = a(p);
            this.b = b(p);
        }
    }

    @Override
    public INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace) {
        //https://arxiv.org/pdf/1706.02515.pdf pg6
        // "...we propose “alpha dropout”, that randomly sets inputs to α'"
        // "The affine transformation a(xd + α'(1−d))+b allows to determine parameters a and b such that mean and
        // variance are kept to their values"

        double pValue;
        if(pSchedule != null){
            pValue = pSchedule.valueAt(lastPValue, iteration, epoch);
        } else {
            pValue = p;
        }

        if(pValue != lastPValue){
            a = a(pValue);
            b = b(pValue);
        }
        lastPValue = pValue;

        INDArray result = inPlace ? inputActivations : inputActivations.dup(inputActivations.ordering());
        Nd4j.getExecutioner().exec(new AlphaDropOut(result, p, a, alphaPrime, b));

        return result;
    }

    @Override
    public AlphaDropout clone() {
        return new AlphaDropout(p, pSchedule == null ? null : pSchedule.clone(), alpha, lambda);
    }

    private double a(double p){
        return Math.sqrt(p + alphaPrime*alphaPrime * p * (1-p));
    }

    private double b(double p){
        return -a(p) * (1-p)*alphaPrime;
    }
}
