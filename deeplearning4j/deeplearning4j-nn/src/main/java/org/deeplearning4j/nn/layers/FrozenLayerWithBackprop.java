/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.util.OneTimeLogger;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Frozen layer freezes parameters of the layer it wraps, but allows the backpropagation to continue.
 *
 * @author Ugljesa Jovanovic (jovanovic.ugljesa@gmail.com)
 */

@Slf4j
public class FrozenLayerWithBackprop extends BaseWrapperLayer {

    private boolean logUpdate = false;
    private boolean logFit = false;
    private boolean logTestMode = false;
    private boolean logGradient = false;

    private Gradient zeroGradient;

    public FrozenLayerWithBackprop(final Layer insideLayer) {
        super(insideLayer);
        this.zeroGradient = new DefaultGradient(insideLayer.params());
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
        INDArray backpropEpsilon = underlying.backpropGradient(epsilon, workspaceMgr).getSecond();
        //backprop might have already changed the gradient view (like BaseLayer and BaseOutputLayer do)
        //so we want to put it back to zeroes
        INDArray gradientView = underlying.getGradientsViewArray();
        if(gradientView != null){
            gradientView.assign(0);
        }
        return new Pair<>(zeroGradient, backpropEpsilon);
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
        return new FrozenLayerWithBackprop(underlying.clone());
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
        underlying.setBackpropGradientsViewArray(gradients);
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
            OneTimeLogger.info(log, "Frozen layers cannot be fit, but backpropagation will continue.Warning will be issued only once per instance");
            logFit = true;
        }
    }

    @Override
    public void applyConstraints(int iteration, int epoch) {
        //No-op
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


