package org.deeplearning4j.nn.updater;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.schedule.ISchedule;

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
        private final int paramOffsetStart;
        private final int paramOffsetEnd;
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
            String varName = varState.getParamName();
            gradientUpdater = varState.getLayer().conf().getLayer().getIUpdaterByParam(varName).instantiate(updaterView,
                            updaterViewRequiresInitialization); //UpdaterUtils.getGradientUpdater(varState.getLayer(), varState.getParamName());
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

    public GradientUpdater getGradientUpdater() {
        if (gradientUpdater == null) {
            init();
        }
        return gradientUpdater;
    }

    /**
     * Update the gradient for this block
     *
     * @param iteration The current iteration (i.e., total number of parameter updates so far)
     */
    public void update(int iteration) {
        update(iteration, false, gradientView, null);
    }

    public void updateExternalGradient(int iteration, INDArray fullNetworkGradientView,
                    INDArray fullNetworkParamsArray) {
        //Extract the relevant subset from the external network
        update(iteration, true, fullNetworkGradientView, fullNetworkParamsArray);
    }

    private void update(int iteration, boolean externalGradient, INDArray fullNetworkGradientView,
                    INDArray fullNetworkParamsArray) {
        //Initialize the updater, if necessary
        if (gradientUpdater == null) {
            init();
        }

        INDArray blockGradViewArray;
        if (externalGradient) {
            blockGradViewArray = fullNetworkGradientView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(paramOffsetStart, paramOffsetEnd));
        } else {
            blockGradViewArray = gradientView;
        }

        //First: Pre-apply gradient clipping etc: some are done on a per-layer basis
        //Therefore: it's already done by this point, in MultiLayerUpdater or ComputationGraphUpdater

        //Second: apply learning rate policy. Note that by definition we have the same LR policy for every single
        // variable in the block
        Layer l0 = layersAndVariablesInBlock.get(0).getLayer();
        if (!(l0.conf().getLayer() instanceof BaseLayer)) {
            //No params for this layer
            return;
        }
        BaseLayer baseLayer = (BaseLayer) l0.conf().getLayer();
        String firstParam = layersAndVariablesInBlock.get(0).getParamName();
        boolean isBias = l0.conf().getLayer().initializer().isBiasParam(firstParam);
        ISchedule lrSchedule;
        if(isBias){
            //TODO should this
            lrSchedule = l0.conf().getLearningRateSchedule();
        } else {

        }

        if (lrPolicy != LearningRatePolicy.None || baseLayer.getIUpdater() instanceof Nesterovs) {
            applyLrDecayPolicy(lrPolicy, iteration);
        }

        //Apply the updater itself
        gradientUpdater.applyUpdater(blockGradViewArray, iteration);

        //Post apply: l1 and l2 by params
        for (ParamState p : layersAndVariablesInBlock) {
            INDArray paramView;
            INDArray gradView;
            if (externalGradient) {
                paramView = fullNetworkParamsArray.get(NDArrayIndex.point(0),
                                NDArrayIndex.interval(p.getParamOffsetStart(), p.getParamOffsetEnd()));
                gradView = fullNetworkGradientView.get(NDArrayIndex.point(0),
                                NDArrayIndex.interval(p.getParamOffsetStart(), p.getParamOffsetEnd()));
            } else {
                //Standard case
                paramView = p.getParamView();
                gradView = p.getGradView();
            }
            postApply(p.getLayer(), p.getParamName(), gradView, paramView);
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
        if (l2 > 0) {
            //This can be an axpy op, saving an allocation...
            //gradientView += params * l2           i.e., dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
            //Equivalent to gradientView.addi(paramsView.mul(conf.getL2ByParam(paramName)));
            int length = gradientView.length();
            Nd4j.getBlasWrapper().level1().axpy(length, l2, paramsView, gradientView);
        }
        if (conf.getL1ByParam(paramName) > 0) {
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

        if (!(conf.getLayer() instanceof BaseLayer)) {
            //No params
            return;
        }

        BaseLayer baseLayer = (BaseLayer) conf.getLayer();

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
                if (baseLayer.getLearningRateSchedule().containsKey(iteration)) {
                    newLr = baseLayer.getLearningRateSchedule().get(iteration);
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

        //Handle momentum schedules. Given the new updater design, this change is purely cosmetic
        double newMomentum = 0.0;
        if (baseLayer.getIUpdater() instanceof Nesterovs) {
            if (baseLayer.getMomentumSchedule() != null && baseLayer.getMomentumSchedule().containsKey(iteration)) {
                newMomentum = baseLayer.getMomentumSchedule().get(iteration);
            } else {
                newMomentum = baseLayer.getMomentum();
            }
        }

        //Need to set the LR for *all* variables in the Updater block. All variables (by definition of being in the
        // same block) share the same LR schedule
        for (ParamState vs : layersAndVariablesInBlock) {
            vs.getLayer().conf().setLearningRateByParam(vs.getParamName(), newLr);
            if (((BaseLayer) layer.conf().getLayer()).getIUpdater() instanceof Nesterovs) {
                ((BaseLayer) vs.getLayer().conf().getLayer()).setMomentum(newMomentum);
            }
        }

        //Apply the new LR according to the schedule.
        //Note: momentum schedules are applied internally in the Nesterov config object applySchedules method
        gradientUpdater.getConfig().applySchedules(iteration, newLr);
    }
}
