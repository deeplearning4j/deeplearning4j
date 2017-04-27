package org.deeplearning4j.nn.updater;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * UpdaterBlock: used in {@link BaseMultiLayerUpdater}, this class implements updating (i.e., Adam, RMSProp, Momentum,
 * etc) across multiple contiguous layers/parameters, as described in the {@link BaseMultiLayerUpdater} javadoc.
 *
 * @author Alex Black
 */
@Data
public class UpdaterBlock {
    private int paramOffsetStart;
    private int paramOffsetEnd;
    private int updaterViewOffsetStart;
    private int updaterViewOffsetEnd;
    private List<ParamState> layersAndVariablesInBlock = new ArrayList<>();

    private INDArray updaterView;
    private INDArray gradientView;
    private boolean updaterViewRequiresInitialization;

    private GradientUpdater gradientUpdater;


    @AllArgsConstructor
    @Data
    public static class ParamState {
        private final Layer layer;
        private final String paramName;
        private final INDArray paramView;
        private final INDArray gradView;
    }

    /**
     * @param paramOffsetStart          Start offset of the parameters in this block (relative to overall net params
     *                                  view array)
     * @param paramOffsetEnd            End offset of the parameters in this block (relative to overall net params
     *                                  view array)
     * @param updaterViewOffsetStart    Start offset of the updater state array in this block (relative to overall net
     *                                  updater state view array)
     * @param updaterViewOffsetEnd      End offset of the updater state array in this block (relative to overall net
     *                                  updater state view array)
     * @param layersAndVariablesInBlock List of layers and variables in this updater block. By definition, all layers
     *                                  and variables in this list <i>must</i> have an identical updater configuration.
     */
    public UpdaterBlock(int paramOffsetStart, int paramOffsetEnd, int updaterViewOffsetStart, int updaterViewOffsetEnd,
                    List<ParamState> layersAndVariablesInBlock) {
        this.paramOffsetStart = paramOffsetStart;
        this.paramOffsetEnd = paramOffsetEnd;
        this.updaterViewOffsetStart = updaterViewOffsetStart;
        this.updaterViewOffsetEnd = updaterViewOffsetEnd;
        this.layersAndVariablesInBlock = layersAndVariablesInBlock;
    }

    public void init() {
        if (gradientUpdater == null) {
            ParamState varState = layersAndVariablesInBlock.get(0);
            gradientUpdater = UpdaterUtils.getGradientUpdater(varState.getLayer(), varState.getParamName());
            if (updaterView != null) {
                //May be null for SGD and no-op updaters
                int[] gradientViewShape = gradientView.shape();
                gradientUpdater.setStateViewArray(updaterView, gradientViewShape, 'c',
                                updaterViewRequiresInitialization);
            }
        }
    }

    public boolean isPretrainUpdaterBlock() {
        //All in block should be the same layer, and all be pretrain params
        ParamState vs = layersAndVariablesInBlock.get(0);
        return vs.getLayer().conf().getLayer().isPretrainParam(vs.getParamName());
    }

    public boolean skipDueToPretrainConfig() {
        if (!isPretrainUpdaterBlock())
            return false;
        ParamState vs = layersAndVariablesInBlock.get(0);
        return !vs.getLayer().conf().isPretrain(); //Skip if not pretrain
    }

    /**
     * Update the gradient for this block
     *
     * @param iteration The current iteration (i.e., total number of parameter updates so far)
     */
    public void update(int iteration) {
        //Initialize the updater, if necessary
        if (gradientUpdater == null) {
            init();
        }

        //First: Pre-apply gradient clipping etc: some are done on a per-layer basis
        //Therefore: it's already done by this point, in MultiLayerUpdater or ComputationGraphUpdater

        //Second: apply learning rate policy. Note that by definition we have the same LR policy for every single
        // variable in the block
        Layer l0 = layersAndVariablesInBlock.get(0).getLayer();
        LearningRatePolicy lrPolicy = l0.conf().getLearningRatePolicy();
        if (lrPolicy != LearningRatePolicy.None
                        || l0.conf().getLayer().getUpdater() == org.deeplearning4j.nn.conf.Updater.NESTEROVS) {
            applyLrDecayPolicy(lrPolicy, iteration);
        }

        //Apply the updater itself
        gradientUpdater.getGradient(gradientView, iteration);

        //Post apply: l1 and l2 by params
        for (ParamState p : layersAndVariablesInBlock) {
            postApply(p.getLayer(), p.getParamName(), p.getGradView(), p.getParamView());
        }
    }

    /**
     * Apply L1 and L2 regularization, if necessary. Note that L1/L2 may differ for different layers in the same block
     *
     * @param layer        The layer to apply L1/L2 to
     * @param paramName    Parameter name in the given layer
     * @param gradientView Gradient view array for the layer + param
     * @param paramsView   Parameter view array for the layer + param
     */
    public void postApply(Layer layer, String paramName, INDArray gradientView, INDArray paramsView) {
        NeuralNetConfiguration conf = layer.conf();

        //TODO: do this for multiple contiguous params/layers (fewer, larger ops)

        double l2 = conf.getL2ByParam(paramName);
        if (conf.isUseRegularization() && l2 > 0) {
            //This can be an axpy op, saving an allocation...
            //gradientView += params * l2           i.e., dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
            //Equivalent to gradientView.addi(paramsView.mul(conf.getL2ByParam(paramName)));
            int length = gradientView.length();
            Nd4j.getBlasWrapper().level1().axpy(length, l2, paramsView, gradientView);
        }
        if (conf.isUseRegularization() && conf.getL1ByParam(paramName) > 0) {
            gradientView.addi(Transforms.sign(paramsView, true).muli(conf.getL1ByParam(paramName)));
        }
    }

    /**
     * Apply learning rate decay, based on the configuration
     *
     * @param decay     Learning rate schedule enumeration
     * @param iteration Current iteration
     */
    public void applyLrDecayPolicy(LearningRatePolicy decay, int iteration) {
        Layer layer = layersAndVariablesInBlock.get(0).getLayer();
        String variable = layersAndVariablesInBlock.get(0).getParamName();

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
                if (iteration > 1 && conf.getLrPolicySteps() % iteration == 0) {
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
                if (conf.getLayer().getLearningRateSchedule().containsKey(iteration)) {
                    newLr = conf.getLayer().getLearningRateSchedule().get(iteration);
                } else {
                    newLr = lr;
                }
                break;
            case None:
            case Score:
                newLr = lr;
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
        for (ParamState vs : layersAndVariablesInBlock) {
            vs.getLayer().conf().setLearningRateByParam(vs.getParamName(), newLr);
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
