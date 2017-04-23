package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.AddOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Nesterov's momentum.
 * Keep track of the previous layer's gradient
 * and use it as a way of updating the gradient.
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
@Slf4j
public class Nesterovs implements Serializable, GradientUpdater {
    public static final double DEFAULT_NESTEROV_MOMENTUM = 0.9;

    private double momentum = DEFAULT_NESTEROV_MOMENTUM;
    private volatile INDArray v;
    private double learningRate = 0.1;

    private char gradientReshapeOrder;

    @Override
    public int stateSizeForInputSize(int inputSize) {
        return inputSize;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize) {
        if (!viewArray.isRowVector())
            throw new IllegalArgumentException("Invalid input: expect row vector input");
        if (initialize)
            viewArray.assign(0);

        this.v = viewArray;

        //Reshape to match the expected shape of the input gradient arrays
        this.v = Shape.newShapeNoCopy(this.v, gradientShape, gradientOrder == 'f');
        if (v == null)
            throw new IllegalStateException("Could not correctly reshape gradient view array");
        this.gradientReshapeOrder = gradientOrder;
    }

    public Nesterovs(double momentum, double learningRate) {
        this.momentum = momentum;
        this.learningRate = learningRate;
    }

    public Nesterovs(double momentum) {
        this.momentum = momentum;

    }

    @Override
    public void update(Object... args) {
        if (args.length > 0) {
            learningRate = (Double) args[0];
            momentum = (Double) args[1];
        }

    }


    /**
     * Get the nesterov update
     *
     * @param gradient  the gradient to get the update for
     * @param iteration
     * @return
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if (v == null)
            throw new IllegalStateException("Updater has not been initialized with view state");

        //reference https://cs231n.github.io/neural-networks-3/#sgd 2nd equation
        //DL4J default is negative step function thus we flipped the signs:
        // x += mu * v_prev + (-1 - mu) * v
        //i.e., we do params -= updatedGradient, not params += updatedGradient

        //v = mu * v - lr * gradient
        INDArray vPrev = v.dup(gradientReshapeOrder);
        v.muli(momentum).subi(gradient.dup(gradientReshapeOrder).muli(learningRate));              //Modify state array in-place

        /*
        Next line is equivalent to:
        INDArray ret = vPrev.muli(momentum).addi(v.mul(-momentum - 1));
        gradient.assign(ret);
        */
        Nd4j.getExecutioner().exec(new AddOp(vPrev.muli(momentum), v.mul(-momentum - 1), gradient));

        return gradient;
    }
}
