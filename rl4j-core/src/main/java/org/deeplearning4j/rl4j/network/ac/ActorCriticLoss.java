package org.deeplearning4j.rl4j.network.ac;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonInclude;

/**
 *
 * Custom loss function required for Actor-Critic methods:
 * <pre>
 * L = sum_i advantage_i * log( probability_i )
 * </pre>
 * It is very similar to the Multi-Class Cross Entropy loss function.
 *
 * @author saudet
 * @see LossMCXENT
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ActorCriticLoss implements ILossFunction {

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true).addi(1e-5);
        INDArray scoreArr = Transforms.log(output, false).muli(labels);

        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr;
    }

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
                    boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        double score = -scoreArr.sumNumber().doubleValue();
        return average ? score / scoreArr.size(0) : score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1).muli(-1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true).addi(1e-5);
        INDArray dLda = output.rdivi(labels).negi();
        INDArray grad = activationFn.backprop(preOutput, dLda).getFirst();

        if (mask != null) {
            LossUtil.applyMask(grad, mask);
        }
        return grad;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                    INDArray mask, boolean average) {
        return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                        computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String toString() {
        return "ActorCriticLoss()";
    }

    @Override
    public String name() {
        return toString();
    }
}
