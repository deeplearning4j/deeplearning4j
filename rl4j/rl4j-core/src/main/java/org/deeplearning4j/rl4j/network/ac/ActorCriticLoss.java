package org.deeplearning4j.rl4j.network.ac;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;

/**
 *
 * Custom loss function required for Actor-Critic methods:
 * <pre>
 * L = sum_i advantage_i * log( probability_i ) + entropy( probability )
 * </pre>
 * It is very similar to the Multi-Class Cross Entropy loss function.
 *
 * @author saudet
 * @see LossMCXENT
 */
@EqualsAndHashCode
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ActorCriticLoss implements ILossFunction {

    public static final double BETA = 0.01;

    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true).addi(1e-5);
        INDArray logOutput = Transforms.log(output, true);
        INDArray entropy = output.muli(logOutput);
        INDArray scoreArr = logOutput.muli(labels).subi(entropy.muli(BETA));

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
        INDArray logOutput = Transforms.log(output, true);
        INDArray entropyDev = logOutput.addi(1);
        INDArray dLda = output.rdivi(labels).subi(entropyDev.muli(BETA)).negi();
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
