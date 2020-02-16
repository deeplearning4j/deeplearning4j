package org.nd4j.linalg.lossfunctions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.Map;


public abstract class SameDiffLoss implements ILossFunction {

    SameDiff sd = SameDiff.create();


    protected SameDiffLoss() {


        SDVariable layerInput =  sd.placeHolder("layerInput", DataType.FLOAT ,-1);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT ,-1);
        this.defineLoss(sd, layerInput, labels);


    }


    public abstract SDVariable defineLoss(SameDiff sd, SDVariable layerInput, SDVariable labels);

    /**
     * Compute the score (loss function value) for the given inputs.
     * @param labels       Label/expected preOutput
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/preOutput) or not   @return Loss function value
     */
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {

            // The score overall consists of the
            // sum of the negative log likelihoods for each
            // of the individual labels.



        INDArray scoreArr = computeScoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();
            if (average) {
                score /= scoreArr.size(0);
            }
            return score;
        }






    /**
     * Compute the score (loss function value) for each example individually.
     * For input [numExamples,nOut] returns scores as a column vector: [numExamples,1]
     * @param labels       Labels/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         @return Loss function value for each example; column vector
     */
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask){

        Preconditions.checkArgument((labels.size(1) != preOutput.size(1)),"Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer\n" +
                "number of outputs (nOut = " + preOutput.size(1) + ")");


        INDArray scoreArr;
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        sd.var("out",output);
        scoreArr = output.rsubi(labels).divi(labels);
        scoreArr.muli(100.0 / labels.size(1));



        if (mask != null) {
            LossUtil.applyMask(scoreArr, mask);
        }
        return scoreArr.sum(1);


    }


    /**
     * Compute the gradient of the loss function with respect to the inputs: dL/dOutput
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network), before the activation function is applied
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @return Gradient dL/dPreOut
     */
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {



        Map<String,INDArray> grads = sd.calculateGradients(Collections.singletonMap("out", activationFn.getActivation(preOutput.dup(), true)));

        if (mask != null) {
            LossUtil.applyMask(grads.get("out"), mask);
        }
        return grads.get("out");
    }

    /**
     * Compute both the score (loss function value) and gradient. This is equivalent to calling {@link #computeScore(INDArray, INDArray, IActivation, INDArray, boolean)}
     * and {@link #computeGradient(INDArray, INDArray, IActivation, INDArray)} individually
     *
     * @param labels       Label/expected output
     * @param preOutput    Output of the model (neural network)
     * @param activationFn Activation function that should be applied to preOutput
     * @param mask         Mask array; may be null
     * @param average      Whether the score should be averaged (divided by number of rows in labels/output) or not
     * @return The score (loss function value) and gradient
     */


    public  Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
                                                   INDArray mask, boolean average) {

        Pair<Double, INDArray> GradientAndScore = new Pair<>();
        GradientAndScore.setFirst(this.computeScore(labels, preOutput, activationFn, mask, average));
        GradientAndScore.setSecond(this.computeGradient(labels, preOutput, activationFn, mask));

        return GradientAndScore;
    }

    public String name() {
        return toString();
    }

}




