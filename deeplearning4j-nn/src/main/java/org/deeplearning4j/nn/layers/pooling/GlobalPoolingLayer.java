package org.deeplearning4j.nn.layers.pooling;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Alex on 17/01/2017.
 */
public class GlobalPoolingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer> {

    private static final int[] DEFAULT_TIMESERIES_POOL_DIMS = new int[]{2};
    private static final int[] DEFAULT_CNN_POOL_DIMS = new int[]{2,3};


    private final int[] poolingDimensions;
    private final boolean collapseDimensions;
    private final PoolingType poolingType;

    public GlobalPoolingLayer(NeuralNetConfiguration conf) {
        super(conf);

        org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer) conf.getLayer();

        poolingDimensions = layerConf.getPoolingDimensions();
        collapseDimensions = layerConf.isCollapseDimensions();
        poolingType = layerConf.getPoolingType();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.SUBSAMPLING;
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null){
            throw new IllegalStateException("Cannot perform forward pass: input not set for layer " + layerNameAndIndex());
        }

        int[] poolDim = null;
        if(input.rank() == 3){
            //TODO validation on pooling dimensions

            if(poolingDimensions == null){
                //Use default pooling dimensions;
                poolDim = DEFAULT_TIMESERIES_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }

        } else if(input.rank() == 4){
            //CNN activations
            if(poolingDimensions == null){
                //Use default pooling dimensions;
                poolDim = DEFAULT_CNN_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }
        }

        INDArray reduced2d;
        if(maskArray == null){
            //Standard 'full array' global pooling op
            reduced2d = activateHelperFullArray(input, poolDim);
        } else {
            throw new UnsupportedOperationException("Not yet implemeted");
        }

        if(collapseDimensions){
            return reduced2d;
        } else {
            throw new UnsupportedOperationException("Not yet implemeted");
        }
    }

    private INDArray activateHelperFullArray(INDArray inputArray, int[] poolDim){
        switch (poolingType){
            case MAX:
                return input.max(poolDim);
            case AVG:
                return input.mean(poolDim);
            case SUM:
                return input.sum(poolDim);
            case PNORM:
                int pnorm = layerConf().getPnorm();

                throw new UnsupportedOperationException("Not yet implemeted");
            default:
                throw new RuntimeException("Unknown or not supported pooling type: " + poolingType);
        }
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

        Gradient retGradient = new DefaultGradient();   //Empty: no params

        int[] poolDim = null;
        if(input.rank() == 3){
            //TODO validation on pooling dimensions


            if(poolingDimensions == null){
                //Use default pooling dimensions;
                poolDim = DEFAULT_TIMESERIES_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }

        } else if(input.rank() == 4){
            //CNN activations

            //CNN activations
            if(poolingDimensions == null){
                //Use default pooling dimensions;
                poolDim = DEFAULT_CNN_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }
        }

        INDArray epsilonNd;
        if(maskArray == null){
            //Standard 'full array' global pooling op
            epsilonNd = epsilonHelperFullArray(input, epsilon, poolDim);
        } else {
            throw new UnsupportedOperationException("Not yet implemeted");
        }

        if(collapseDimensions){
            return new Pair<>(retGradient, epsilonNd);
        } else {
            throw new UnsupportedOperationException("Not yet implemeted");
        }
    }

    private INDArray epsilonHelperFullArray(INDArray inputArray, INDArray epsilon, int[] poolDim){

        //Broadcast: occurs on the remaining dimensions, after the pool dimensions have been removed.
        //TODO find a more efficient way to do this
        int[] broadcastDims = new int[inputArray.rank()-poolDim.length];
        int count =0;
        for( int i=0; i<inputArray.rank(); i++ ){
            if(ArrayUtils.contains(poolDim, i)) continue;
            broadcastDims[count++] = i;
        }

        switch (poolingType){
            case MAX:
                INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(inputArray.dup(), poolDim));
                return Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(isMax,epsilon,isMax, broadcastDims));
            case AVG:
                //if out = avg(in,dims) then dL/dIn = 1/N * dL/dOut
                int n = 1;
                for( int d : poolDim ){
                    n *= inputArray.size(d);
                }
                INDArray ret = Nd4j.create(inputArray.shape());
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(ret,epsilon,ret, broadcastDims));
                ret.divi(n);

                return ret;
            case SUM:
                INDArray retSum = Nd4j.create(inputArray.shape());
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(retSum,epsilon,retSum, broadcastDims));
                return retSum;
            case PNORM:
                int pnorm = layerConf().getPnorm();

                throw new UnsupportedOperationException("Not yet implemeted");
            default:
                throw new RuntimeException("Unknown or not supported pooling type: " + poolingType);
        }
    }

}
