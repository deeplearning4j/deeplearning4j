package org.deeplearning4j.nn.updater;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Alex on 14/04/2017.
 */
@Data
public class UpdaterBlock {
    private int paramOffsetStart;
    private int paramOffsetEnd;
    private int updaterViewOffsetStart;
    private int updaterViewOffsetEnd;
    private List<VarState> layersAndVariablesInBlock = new ArrayList<>();

    private INDArray updaterView;
    private INDArray gradientView;
    private boolean updaterViewRequiresInitialization;

    private GradientUpdater gradientUpdater;


    @AllArgsConstructor @Data
    public static class VarState {
        private final Layer layer;
        private final String varName;
        private final INDArray paramView;
        private final INDArray gradView;
    }

    public UpdaterBlock(int paramOffsetStart, int paramOffsetEnd, int updaterViewOffsetStart, int updaterViewOffsetEnd,
                        List<VarState> layersAndVariablesInBlock) {
        this.paramOffsetStart = paramOffsetStart;
        this.paramOffsetEnd = paramOffsetEnd;
        this.updaterViewOffsetStart = updaterViewOffsetStart;
        this.updaterViewOffsetEnd = updaterViewOffsetEnd;
        this.layersAndVariablesInBlock = layersAndVariablesInBlock;
    }

    public void update(int iteration){

        //Initialize the updater, if necessary
        if(gradientUpdater == null){
            VarState varState = layersAndVariablesInBlock.get(0);
            gradientUpdater = UpdaterUtils.getGradientUpdater(varState.getLayer(), varState.getVarName());
            if(updaterView != null) {
                //May be null for SGD and no-op updaters
                int[] gradientViewShape = gradientView.shape();
                System.out.println(Arrays.toString(gradientViewShape));
                gradientUpdater.setStateViewArray(updaterView, gradientViewShape, 'c', updaterViewRequiresInitialization);
            }
        }

        //First: Pre-apply gradient clipping etc: some are done on a per-layer basis
        //Therefore: it's already done by this point, in MultiLayerUpdater or ComputationGraphUpdater

        //Second: apply learning rate policy. Note that by definition we have the same LR policy for every single
        // variable in the block
        Layer l0 = layersAndVariablesInBlock.get(0).getLayer();
        LearningRatePolicy lrPolicy = l0.conf().getLearningRatePolicy();
        if (lrPolicy != LearningRatePolicy.None ||
                l0.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS) {
            applyLrDecayPolicy(lrPolicy, iteration);
        }

        //Apply the updater itself
        gradientUpdater.getGradient(gradientView, iteration);

        //Post apply: l1 and l2 by params
        for(VarState p : layersAndVariablesInBlock){
            postApply(p.getLayer(), p.getVarName(), p.getGradView(), p.getParamView() );
        }
    }

    public void postApply(Layer layer, String paramName, INDArray gradientView, INDArray paramsView) {
        NeuralNetConfiguration conf = layer.conf();

        //TODO: do this for multiple contiguous params/layers (fewer, larger ops)

        double l2 = conf.getL2ByParam(paramName);
        if (conf.isUseRegularization() && l2 > 0){
            //This can be an axpy op, saving an allocation...
            //gradientView += params * l2           i.e., dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
            //Equivalent to gradientView.addi(paramsView.mul(conf.getL2ByParam(paramName)));
            int length = gradientView.length();
            Nd4j.getBlasWrapper().level1().axpy(length, l2, paramsView, gradientView );
        }
        if (conf.isUseRegularization() && conf.getL1ByParam(paramName) > 0) {
            gradientView.addi(Transforms.sign(paramsView).muli(conf.getL1ByParam(paramName)));
        }
    }

    public void applyLrDecayPolicy(LearningRatePolicy decay, int iteration) {
        Layer layer = layersAndVariablesInBlock.get(0).getLayer();
        String variable = layersAndVariablesInBlock.get(0).getVarName();

        NeuralNetConfiguration conf = layer.conf();
        double decayRate = layer.conf().getLrPolicyDecayRate();
        double lr = conf.getLearningRateByParam(variable);

        double newLr;
        switch (decay) {
            case Exponential:
                newLr = lr * Math.pow(decayRate, iteration);
                break;
            case Inverse:
                newLr = lr / Math.pow((1 + decayRate * iteration), conf.getLrPolicyPower());
                break;
            case Step:
                newLr = lr * Math.pow(decayRate, Math.floor(iteration / conf.getLrPolicySteps()));
                break;
            case TorchStep:
                if (iteration > 1 && conf.getLrPolicySteps() % iteration == 0){
                    newLr = lr * decayRate;
                } else {
                    newLr = lr;
                }
                break;
            case Poly:
                newLr = lr * Math.pow((1 - ((double) iteration) / conf.getNumIterations()), conf.getLrPolicyPower());
                break;
            case Sigmoid:
                newLr = lr / (1 + Math.exp(-decayRate * (iteration - conf.getLrPolicySteps())));
                break;
            case Schedule:
                if (conf.getLayer().getLearningRateSchedule().containsKey(iteration)){
                    newLr = conf.getLayer().getLearningRateSchedule().get(iteration);
                } else {
                    newLr = lr;
                }
                break;
            default:
                throw new RuntimeException("Unknown Learning rate decay value: " + decay);
        }

        //Handle momentum schedules
        double newMomentum = 0.0;
        if (layer.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS) {
            if (conf.getLayer().getMomentumSchedule().containsKey(iteration)) {
                newMomentum = conf.getLayer().getMomentumSchedule().get(iteration);
            } else {
                newMomentum = conf.getLayer().getMomentum();
            }
        }

        //Need to set the LR for *all* variables in the Updater block. All variables (by definition of being in the
        // same block) share the same LR schedule
        for(VarState vs : layersAndVariablesInBlock){
            vs.getLayer().conf().setLearningRateByParam(vs.getVarName(), newLr);
            if (layer.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS) {
                vs.getLayer().conf().getLayer().setMomentum(newMomentum);
            }
        }

        if (layer.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS) {
            gradientUpdater.update(newLr, newMomentum);
        } else {
            gradientUpdater.update(newLr);
        }
    }
}
