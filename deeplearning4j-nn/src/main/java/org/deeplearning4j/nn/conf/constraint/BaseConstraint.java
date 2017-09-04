package org.deeplearning4j.nn.conf.constraint;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.Set;


@AllArgsConstructor
@EqualsAndHashCode
@Data
public abstract class BaseConstraint implements LayerConstraint {
    public static final double DEFAULT_EPSILON = 1e-6;

    protected double epsilon = 1e-6;
    protected int[] dimensions;

    protected BaseConstraint(){
        //No arg for json ser/de
    }

    protected BaseConstraint(int... dimensions){
        this(DEFAULT_EPSILON, dimensions);
    }


    @Override
    public void applyConstraint(Layer layer, int iteration, int epoch, boolean hasBiasConstraint,
                                boolean hasWeightConstraint, Set<String> paramNames) {
        Map<String,INDArray> paramTable = layer.paramTable();
        if(paramTable == null || paramTable.isEmpty() ){
            return;
        }

        ParamInitializer i = layer.conf().getLayer().initializer();
        for(Map.Entry<String,INDArray> e : paramTable.entrySet()){
            if(hasWeightConstraint && i.isWeightParam(e.getKey())
                    || hasBiasConstraint && i.isBiasParam(e.getKey())){
                apply(e.getValue());
            }
        }
    }

    public abstract void apply(INDArray param);

    public abstract BaseConstraint clone();

    public static int[] getBroadcastDims(int[] reduceDimensions, int rank){
        int[] out = new int[rank-reduceDimensions.length];
        int outPos = 0;
        for( int i=0; i<rank; i++ ){
            if(!ArrayUtils.contains(reduceDimensions, i)){
                out[outPos++] = i;
            }
        }
        return out;
    }
}
