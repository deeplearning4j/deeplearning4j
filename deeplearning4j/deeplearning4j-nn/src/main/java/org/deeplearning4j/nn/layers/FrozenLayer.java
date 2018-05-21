package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.util.OneTimeLogger;

import java.util.Collection;
import java.util.Map;

/**
 * For purposes of transfer learning
 * A frozen layers wraps another dl4j layer within it.
 * The params of the layer within it are "frozen" or in other words held constant
 * During the forward pass the frozen layer behaves as the layer within it would during test regardless of the training/test mode the network is in.
 * Backprop is skipped since parameters are not be updated.
 * @author susaneraly
 */
@Slf4j
public class FrozenLayer extends BaseWrapperLayer {

    private boolean logUpdate = false;
    private boolean logFit = false;
    private boolean logTestMode = false;
    private boolean logGradient = false;
    private Gradient zeroGradient;

    public FrozenLayer(Layer insideLayer) {
        super(insideLayer);
        if (insideLayer instanceof OutputLayer) {
            throw new IllegalArgumentException("Output Layers are not allowed to be frozen " + layerId());
        }
        this.zeroGradient = new DefaultGradient(insideLayer.params());
        if (insideLayer.paramTable() != null) {
            for (String paramType : insideLayer.paramTable().keySet()) {
                //save memory??
                zeroGradient.setGradientFor(paramType, null);
            }
        }
    }

    @Override
    public void setCacheMode(CacheMode mode) {
        // no-op
    }

    protected String layerId() {
        String name = underlying.conf().getLayer().getLayerName();
        return "(layer name: " + (name == null ? "\"\"" : name) + ", layer index: " + underlying.getIndex() + ")";
    }

    @Override
    public double calcL2(boolean backpropOnlyParams) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropOnlyParams) {
        return 0;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return new Pair<>(zeroGradient, null);
    }
    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        logTestMode(training);
        return underlying.activate(false, workspaceMgr);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        logTestMode(training);
        return underlying.activate(input, false, workspaceMgr);
    }

    @Override
    public Layer clone() {
        OneTimeLogger.info(log, "Frozen layers are cloned as their original versions.");
        return new FrozenLayer(underlying.clone());
    }

    @Override
    public void fit() {
        if (!logFit) {
            OneTimeLogger.info(log, "Frozen layers cannot be fit. Warning will be issued only once per instance");
            logFit = true;
        }
        //no op
    }

    @Override
    public void update(Gradient gradient) {
        if (!logUpdate) {
            OneTimeLogger.info(log, "Frozen layers will not be updated. Warning will be issued only once per instance");
            logUpdate = true;
        }
        //no op
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        if (!logUpdate) {
            OneTimeLogger.info(log, "Frozen layers will not be updated. Warning will be issued only once per instance");
            logUpdate = true;
        }
        //no op
    }
    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        if (!logGradient) {
            OneTimeLogger.info(log,
                            "Gradients for the frozen layer are not set and will therefore will not be updated.Warning will be issued only once per instance");
            logGradient = true;
        }
        underlying.score();
        //no op
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        if (!logGradient) {
            OneTimeLogger.info(log,
                            "Gradients for the frozen layer are not set and will therefore will not be updated.Warning will be issued only once per instance");
            logGradient = true;
        }
        //no-op
    }

    @Override
    public void fit(INDArray data, LayerWorkspaceMgr workspaceMgr) {
        if (!logFit) {
            OneTimeLogger.info(log, "Frozen layers cannot be fit.Warning will be issued only once per instance");
            logFit = true;
        }
    }

    @Override
    public Gradient gradient() {
        return zeroGradient;
    }

    //FIXME
    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        if (!logGradient) {
            OneTimeLogger.info(log,
                            "Gradients for the frozen layer are not set and will therefore will not be updated.Warning will be issued only once per instance");
            logGradient = true;
        }
        return new Pair<>(zeroGradient, underlying.score());
    }

    @Override
    public void applyConstraints(int iteration, int epoch) {
        //No-op
    }

    /**
     * Init the model
     */
    @Override
    public void init() {

    }

    public void logTestMode(boolean training) {
        if (!training)
            return;
        if (logTestMode) {
            return;
        } else {
            OneTimeLogger.info(log,
                            "Frozen layer instance found! Frozen layers are treated as always in test mode. Warning will only be issued once per instance");
            logTestMode = true;
        }
    }

    public void logTestMode(TrainingMode training) {
        if (training.equals(TrainingMode.TEST))
            return;
        if (logTestMode) {
            return;
        } else {
            OneTimeLogger.info(log,
                            "Frozen layer instance found! Frozen layers are treated as always in test mode. Warning will only be issued once per instance");
            logTestMode = true;
        }
    }

    public Layer getInsideLayer() {
        return underlying;
    }
}


