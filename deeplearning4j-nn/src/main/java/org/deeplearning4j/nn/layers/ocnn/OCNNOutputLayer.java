package org.deeplearning4j.nn.layers.ocnn;


import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import static org.deeplearning4j.nn.layers.ocnn.OCNNParamInitializer.R_KEY;
import static org.deeplearning4j.nn.layers.ocnn.OCNNParamInitializer.V_KEY;
import static org.deeplearning4j.nn.layers.ocnn.OCNNParamInitializer.W_KEY;

/**
 * Layer implementation for {@link org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer}
 * See {@link org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer}
 * for details.
 *
 * @author Adam Gibson
 */
public class OCNNOutputLayer extends BaseOutputLayer<org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer> {
    @Setter
    @Getter
    private  IActivation activation = new ActivationReLU();
    private  static IActivation relu = new ActivationReLU();


    private ILossFunction lossFunction;

    public OCNNOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.lossFunction = new OCNNLossFunction();
        org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = (org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) conf.getLayer();
        ocnnOutputLayer.setLossFn(this.lossFunction);
    }

    public OCNNOutputLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = (org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) conf.getLayer();
        ocnnOutputLayer.setLossFn(this.lossFunction);
    }

    @Override
    public INDArray getLabels() {
        return super.getLabels();
    }

    @Override
    public void setLabels(INDArray labels) {
        //no-op
    }


    /** Compute score after labels and input have been set.
     * @param fullNetworkL1 L1 regularization term for the entire network
     * @param fullNetworkL2 L2 regularization term for the entire network
     * @param training whether score should be calculated at train or test time (this affects things like application of
     *                 dropout, etc)
     * @return score (loss function)
     */
    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (input == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        INDArray preOut = preOutput2d(training, workspaceMgr);

        ILossFunction lossFunction = layerConf().getLossFn();

        double score = lossFunction.computeScore(getLabels2d(workspaceMgr, ArrayType.FF_WORKING_MEM), preOut,
                layerConf().getActivationFn(), maskArray,false);
        score += fullNetworkL1 + fullNetworkL2;
        if(conf().isMiniBatch())
            score /= getInputMiniBatchSize();

        this.score = score;

        return score;
    }

    @Override
    public boolean needsLabels() {
        return false;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        Pair<Gradient, INDArray> pair = getGradientsAndDelta(preOutput2d(true, workspaceMgr), workspaceMgr); //Returns Gradient and delta^(this), not Gradient and epsilon^(this-1)
        //150
       int inputShape = (( org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) this.getConf().getLayer()).getNIn();
        INDArray delta = pair.getSecond();
        //4 x 150
        INDArray epsilonNext = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, new int[]{inputShape, delta.length()}, 'f');
        epsilonNext = epsilonNext.assign(delta.broadcast(epsilonNext.shape())).transpose();

        //Normally we would clear weightNoiseParams here - but we want to reuse them for forward + backward + score
        // So this is instead done in MultiLayerNetwork/CompGraph backprop methods

        return new Pair<>(pair.getFirst(), epsilonNext);
    }


    /** Returns tuple: {Grafdient,Delta,Output} given preOut */
    private Pair<Gradient, INDArray> getGradientsAndDelta(INDArray preOut, LayerWorkspaceMgr workspaceMgr) {
        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray labels2d = getLabels2d(workspaceMgr, ArrayType.BP_WORKING_MEM);
        INDArray delta = lossFunction.computeGradient(labels2d, preOut, layerConf().getActivationFn(), maskArray);

        Gradient gradient = new DefaultGradient();
        INDArray vGradView = gradientViews.get(V_KEY);
        double oneDivNu = 1.0 / layerConf().getNu();
        INDArray xTimesV = input.mmul(getParam(V_KEY));
        INDArray derivW = layerConf().getActivationFn()
                .getActivation(xTimesV.dup(),true).negi();
        derivW = delta.isRowVector() ? derivW
                .muliRowVector(delta).mean(0)
                .muli(oneDivNu).addi(getParam(W_KEY)) : derivW
                .muli(delta).mean(0)
                .muli(oneDivNu).addi(getParam(W_KEY));
        gradient.setGradientFor(W_KEY,gradientViews.get(W_KEY).assign(derivW));

        //dG -> sigmoid derivative

        INDArray firstVertDerivV =  layerConf().getActivationFn()
                .backprop(xTimesV.dup(),Nd4j.scalar(1.0))
                .getFirst().muliRowVector(getParam(W_KEY).neg());
        firstVertDerivV = delta.isRowVector() ?
                firstVertDerivV .muliRowVector(delta)
                .reshape('f',input.size(0),1,layerConf().getHiddenSize()) :
                firstVertDerivV.muli(delta)
                        .reshape('f',input.size(0),1,layerConf().getHiddenSize());
        INDArray secondTermDerivV = input.reshape('f',
                input.size(0),getParam(V_KEY).size(0),1);

        int[]  shape = new int[firstVertDerivV.shape().length];
        for(int i = 0; i < firstVertDerivV.rank(); i++) {
            shape[i] = Math.max(firstVertDerivV.size(i),secondTermDerivV.size(i));
        }

        INDArray firstDerivVBroadcast = Nd4j.createUninitialized(shape);

        INDArray mulResult = firstVertDerivV.broadcast(firstDerivVBroadcast);
        int[] bcDims = Shape.getBroadcastDimensions(mulResult.shape(), secondTermDerivV.shape());
        Broadcast.mul(mulResult, secondTermDerivV, mulResult, bcDims);

        INDArray derivV = mulResult
                .mean(0).muli(oneDivNu).addi(getParam(V_KEY));
        gradient.setGradientFor(V_KEY,vGradView.assign(derivV));

        INDArray derivR = Nd4j.scalar(delta.meanNumber()).muli(oneDivNu).addi(-1);
        gradient.setGradientFor(R_KEY,Nd4j.scalar(0.0));

        gradient.setGradientFor(R_KEY,gradientViews.get(R_KEY).assign(derivR));
        clearNoiseWeightParams();

        return new Pair<>(gradient, delta);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        this.input = input;
        return doOutput(training,workspaceMgr);
    }

    /**{@inheritDoc}
     */
    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        float[] decision = examples.data().asFloat();
        for(int i = 0; i < decision.length; i++) {
            if(decision[i] < 0) {
                decision[i] = 0.0f;
            }
            else {
                decision[i] = 1.0f;
            }
        }

        return Nd4j.create(decision);
    }


    @Override
    public Layer.Type type() {
        return Type.FEED_FORWARD;
    }




    @Override
    protected INDArray preOutput2d(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return doOutput(training,workspaceMgr);
    }

    @Override
    protected INDArray getLabels2d(LayerWorkspaceMgr workspaceMgr, ArrayType arrayType) {
        return labels;
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return doOutput(training,workspaceMgr);
    }

    private INDArray doOutput(boolean training,LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        INDArray w = getParamWithNoise(W_KEY,training,workspaceMgr);
        INDArray v = getParamWithNoise(V_KEY,training,workspaceMgr);
        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray first = Nd4j.createUninitialized(input.size(0), v.size(1));
        input.mmuli(v, first);
        INDArray act2d = layerConf().getActivationFn().getActivation(first, training);
        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS,input.size(0));
        act2d.mmuli(w.reshape(w.length()), output);
        this.labels = output;
        return output;
    }




    /**Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkL1 L1 regularization term for the entire network (or, 0.0 to not include regularization)
     * @param fullNetworkL2 L2 regularization term for the entire network (or, 0.0 to not include regularization)
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2, LayerWorkspaceMgr workspaceMgr) {
        //For RNN: need to sum up the score over each time step before returning.

        if (input == null || labels == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        INDArray preOut = preOutput2d(false, workspaceMgr);

        ILossFunction lossFunction = layerConf().getLossFn();
        INDArray scoreArray =
                lossFunction.computeScoreArray(getLabels2d(workspaceMgr, ArrayType.FF_WORKING_MEM), preOut,
                        layerConf().getActivationFn(), maskArray);
        INDArray summedScores = scoreArray.sum(1);

        double l1l2 = fullNetworkL1 + fullNetworkL2;
        if (l1l2 != 0.0) {
            summedScores.addi(l1l2);
        }

        return summedScores;
    }

    public class OCNNLossFunction implements ILossFunction {

        @Override
        public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
            double wSum = Transforms.pow(getParam(W_KEY),2).sumNumber().doubleValue() * 0.5;
            double vSum = Transforms.pow(getParam(V_KEY),2).sumNumber().doubleValue() * 0.5;
            org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer ocnnOutputLayer = (org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer) conf().getLayer();
            INDArray rMeanSub  = relu.getActivation(getParam(R_KEY).sub(preOutput),true);
            double rMean = rMeanSub.meanNumber().doubleValue();
            double rSum = getParam(R_KEY).getDouble(0);
            double nuDiv = (1 / ocnnOutputLayer.getNu()) * rMean;
            double lastTerm = -rSum;
            return (wSum + vSum + nuDiv + lastTerm);
        }

        @Override
        public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
            INDArray r = getParam(R_KEY).sub(preOutput);
            return  r;
        }

        @Override
        public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
            INDArray preAct = getParam(R_KEY).sub(preOutput);
            INDArray target =   relu.backprop(getParam(R_KEY).sub(preOutput),Nd4j.ones(preAct.shape())).getFirst();
            return target;
        }

        @Override
        public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
            //TODO: probably a more efficient way to do this...
            return new Pair<>(computeScore(labels, preOutput, activationFn, mask, average),
                    computeGradient(labels, preOutput, activationFn, mask));
        }

        @Override
        public String name() {
            return "OCNNLossFunction";
        }
    }
}
