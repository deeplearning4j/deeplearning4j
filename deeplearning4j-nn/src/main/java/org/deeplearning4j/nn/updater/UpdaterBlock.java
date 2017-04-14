package org.deeplearning4j.nn.updater;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
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

    public void update(int iteration, int batchSize){

        if(gradientUpdater == null){
            VarState varState = layersAndVariablesInBlock.get(0);
            gradientUpdater = UpdaterUtils.getGradientUpdater(varState.getLayer(), varState.getVarName());
            if(updaterView != null) {
                //May be null for SGD and no-op updaters
                gradientUpdater.setStateViewArray(updaterView, new int[]{1, updaterView.length()}, 'c', true);
            }
        }

        //Pre-apply gradient clipping etc: some are done on a per-layer basis

        //Update LR based on schedules

        gradientUpdater.getGradient(gradientView, iteration);

        //Post apply: l1 and l2 by params, division by minibatch size
        for(VarState p : layersAndVariablesInBlock){
            postApply(p.getLayer(), p.getVarName(), p.getGradView(), p.getParamView(), batchSize );
        }
    }

    public void postApply(Layer layer, String paramName, INDArray gradientView, INDArray paramsView, int miniBatchSize) {
        NeuralNetConfiguration conf = layer.conf();

        if (conf.isUseRegularization() && conf.getL2ByParam(paramName) > 0)
            gradientView.addi(paramsView.mul(conf.getL2ByParam(paramName))); //dC/dw = dC0/dw + lambda/n * w where C0 is pre-l2 cost function
        if (conf.isUseRegularization() && conf.getL1ByParam(paramName) > 0)
            gradientView.addi(Transforms.sign(paramsView).muli(conf.getL1ByParam(paramName)));
        //Done in MultiLayerUpdater.update
//        if (conf.isMiniBatch())
//            gradientView.divi(miniBatchSize);
    }
}
