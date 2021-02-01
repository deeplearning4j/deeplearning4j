/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.updater;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Trainable;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
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
        private final Trainable layer;
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
            gradientUpdater = varState.getLayer().getConfig().getUpdaterByParam(varName).instantiate(updaterView,
                            updaterViewRequiresInitialization); //UpdaterUtils.getGradientUpdater(varState.getLayer(), varState.getParamName());
        }
    }

    public boolean isPretrainUpdaterBlock() {
        //All in block should be the same layer, and all be pretrain params
        ParamState vs = layersAndVariablesInBlock.get(0);
        return vs.getLayer().getConfig().isPretrainParam(vs.getParamName());
    }

    public boolean skipDueToPretrainConfig( boolean isLayerUpdater) {
        if (!isPretrainUpdaterBlock())
            return false;
        return !isLayerUpdater;
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
    public void update(int iteration, int epoch) {
        update(iteration, epoch, false, gradientView, null);
    }

    public void updateExternalGradient(int iteration, int epoch, INDArray fullNetworkGradientView,
                    INDArray fullNetworkParamsArray) {
        //Extract the relevant subset from the external network
        update(iteration, epoch, true, fullNetworkGradientView, fullNetworkParamsArray);
    }

    private void update(int iteration, int epoch, boolean externalGradient, INDArray fullNetworkGradientView,
                    INDArray fullNetworkParamsArray) {
        //Initialize the updater, if necessary
        if (gradientUpdater == null) {
            init();
        }

        INDArray blockGradViewArray;
        if (externalGradient) {
            blockGradViewArray = fullNetworkGradientView.get(NDArrayIndex.interval(0,0,true),
                            NDArrayIndex.interval(paramOffsetStart, paramOffsetEnd));
        } else {
            blockGradViewArray = gradientView;
        }

        //First: Pre-apply gradient clipping etc: some are done on a per-layer basis
        //Therefore: it's already done by this point, in MultiLayerUpdater or ComputationGraphUpdater

        //Second: apply learning rate policy. Note that by definition we have the same LR policy for every single
        // variable in the block
        Trainable l0 = layersAndVariablesInBlock.get(0).getLayer();
        if (l0.numParams() == 0) {
            //No params for this layer
            return;
        }

        //Pre-updater regularization: l1 and l2
        applyRegularizationAllVariables(Regularization.ApplyStep.BEFORE_UPDATER, iteration, epoch, externalGradient, fullNetworkGradientView, fullNetworkParamsArray);

        //Apply the updater itself
        gradientUpdater.applyUpdater(blockGradViewArray, iteration, epoch);

        //Post updater regularization: weight decay
        applyRegularizationAllVariables(Regularization.ApplyStep.POST_UPDATER, iteration, epoch, externalGradient, fullNetworkGradientView, fullNetworkParamsArray);
    }

    protected void applyRegularizationAllVariables(Regularization.ApplyStep applyStep, int iteration, int epoch, boolean externalGradient, INDArray fullNetworkGradientView,
                                                   INDArray fullNetworkParamsArray) {
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

            boolean hasLR = gradientUpdater.getConfig().hasLearningRate();
            double lr = (hasLR ? gradientUpdater.getConfig().getLearningRate(iteration, epoch) : 1.0);
            applyRegularization(applyStep, p.getLayer(), p.getParamName(), gradView, paramView, iteration, epoch, lr);
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
    protected void applyRegularization(Regularization.ApplyStep step, Trainable layer, String paramName, INDArray gradientView, INDArray paramsView, int iter, int epoch, double lr) {
        //TODO: do this for multiple contiguous params/layers (fewer, larger ops)

        List<Regularization> l = layer.getConfig().getRegularizationByParam(paramName);
        if(l != null && !l.isEmpty()){
            for(Regularization r : l){
                if(r.applyStep() == step){
                    r.apply(paramsView, gradientView, lr, iter, epoch);
                }
            }
        }
    }
}
