package org.nd4j.linalg.lossfunctions.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Mean Squared Logarithmic Error loss function: L = 1/N sum_i (log(1+predicted_i) - log(1+actual_i))^2
 *
 * @author Susan Eraly
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
@Getter
public class LossMSLE implements ILossFunction {

    @JsonSerialize(using = RowVectorSerializer.class)
    @JsonDeserialize(using = RowVectorDeserializer.class)
    private final INDArray weights;

    public LossMSLE() {
        this(null);
    }

    /**
     * Mean Squared Logarithmic Error loss function where each the output is (optionally) weighted/scaled by a fixed scalar value.
     * Note that the weights array must be a row vector, of length equal to the labels/output dimension 1 size.
     * A weight vector of 1s should give identical results to no weight vector.
     *
     * @param weights Weights array (row vector). May be null.
     */
    public LossMSLE(INDArray weights) {
        if (weights != null && !weights.isRowVector()) {
            throw new IllegalArgumentException("Weights array must be a row vector");
        }
        this.weights = weights;
    }

    public INDArray scoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr;
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));
        scoreArr = Transforms.log(output.addi(1.0).divi(labels.add(1.0)), false);
        scoreArr = scoreArr.muli(scoreArr).divi(labels.size(1));

        //Weighted loss function
        if (weights != null) {
            if (weights.length() != output.size(1)) {
                throw new IllegalStateException("Weights vector (length " + weights.length() + ") does not match output.size(1)=" + output.size(1));
            }
            scoreArr.muliRowVector(weights);
        }

        if (mask != null) {
            scoreArr.muliColumnVector(mask);
        }
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) score /= scoreArr.size(0);

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, String activationFn, INDArray mask) {
        INDArray output = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()));

        INDArray gradients;
        if ("softmax".equals(activationFn)) {
            INDArray p1 = output.add(1.0);
            INDArray dlda = p1.rdiv(2.0 / labels.size(1));
            INDArray logRatio = Transforms.log(p1.divi(labels.add(1.0)), false);
            dlda.muli(logRatio);

            if (weights != null) {
                dlda.muliRowVector(weights);
            }

            gradients = LossUtil.dLdZsoftmaxi(dlda, output);
        } else {
            INDArray p1 = output.addi(1.0);
            INDArray sigmaPrimeZ = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFn, preOutput.dup()).derivative());
            gradients = sigmaPrimeZ.divi(p1).muli(2.0 / labels.size(1));
            INDArray logRatio = Transforms.log(p1.divi(labels.add(1.0)), false);
            gradients.muli(logRatio);

            //Weighted loss function
            if (weights != null) {
                gradients.muliRowVector(weights);
            }
        }

        if (mask != null) {
            gradients.muliColumnVector(mask);
        }

        return gradients;
    }

    @Override
    public org.apache.commons.math3.util.Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, String activationFn, INDArray mask, boolean average) {
        //TODO: probably a more efficient way to do this...
        //Yes - will implement in round two. Just want to get done now.

        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String toString() {
        if (weights == null) return "LossMSLE()";
        return "LossMSLE(weights=" + weights + ")";
    }

}
