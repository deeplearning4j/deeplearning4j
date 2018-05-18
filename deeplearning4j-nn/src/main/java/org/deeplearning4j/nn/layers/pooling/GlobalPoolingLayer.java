package org.deeplearning4j.nn.layers.pooling;

import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.util.MaskedReductionUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

/**
 * Global pooling layer - used to do pooling over time for RNNs, and 2d pooling for CNNs.<br>
 * Supports the following {@link PoolingType}s: SUM, AVG, MAX, PNORM<br>
 * Global pooling layer can also handle mask arrays when dealing with variable length inputs. Mask arrays are assumed
 * to be 2d, and are fed forward through the network during training or post-training forward pass:<br>
 * - Time series: mask arrays are shape [minibatchSize, maxTimeSeriesLength] and contain values 0 or 1 only<br>
 * - CNNs: mask have shape [minibatchSize, height] or [minibatchSize, width]. Important: the current implementation assumes
 *   that for CNNs + variable length (masking), the input shape is [minibatchSize, channels, height, 1] or
 *   [minibatchSize, channels, 1, width] respectively. This is the case with global pooling in architectures like CNN for
 *   sentence classification.<br>
 * <p>
 * Behaviour with default settings:<br>
 * - 3d (time series) input with shape [minibatchSize, vectorSize, timeSeriesLength] -> 2d output [minibatchSize, vectorSize]<br>
 * - 4d (CNN) input with shape [minibatchSize, channels, height, width] -> 2d output [minibatchSize, channels]<br>
 * <p>
 * Alternatively, by setting collapseDimensions = false in the configuration, it is possible to retain the reduced dimensions
 * as 1s: this gives [minibatchSize, vectorSize, 1] for RNN output, and [minibatchSize, channels, 1, 1] for CNN output.<br>
 *
 * @author Alex Black
 */
public class GlobalPoolingLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer> {

    private static final int[] DEFAULT_TIMESERIES_POOL_DIMS = new int[] {2};
    private static final int[] DEFAULT_CNN_POOL_DIMS = new int[] {2, 3};


    private final int[] poolingDimensions;
    private final boolean collapseDimensions;
    private final PoolingType poolingType;
    private final int pNorm;

    public GlobalPoolingLayer(NeuralNetConfiguration conf) {
        super(conf);

        org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer) conf.getLayer();

        poolingDimensions = layerConf.getPoolingDimensions();
        collapseDimensions = layerConf.isCollapseDimensions();
        poolingType = layerConf.getPoolingType();
        pNorm = layerConf.getPnorm();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
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
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        int[] poolDim;
        if (input.rank() == 3) {
            //TODO validation on pooling dimensions

            if (poolingDimensions == null) {
                //Use default pooling dimensions;
                poolDim = DEFAULT_TIMESERIES_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }

        } else if (input.rank() == 4) {
            //CNN activations
            if (poolingDimensions == null) {
                //Use default pooling dimensions;
                poolDim = DEFAULT_CNN_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }
        } else {
            throw new UnsupportedOperationException("Received rank " + input.rank() + " input (shape = "
                            + Arrays.toString(input.shape()) + "). Only rank 3 (time series) and rank 4 (images"
                            + "/CNN data) are currently supported for global pooling " + layerId());
        }

        INDArray reduced2d;
        if (maskArray == null) {
            //Standard 'full array' global pooling op
            reduced2d = activateHelperFullArray(input, poolDim);
        } else {

            if (input.rank() == 3) {
                //Masked time series

                reduced2d = MaskedReductionUtil.maskedPoolingTimeSeries(poolingType, input, maskArray, pNorm);
            } else if (input.rank() == 4) {
                //Masked convolutions. 4d convolution data, shape [minibatch, channels, h, w]
                //and 2d mask array.
                //Because of this: for now we'll support *masked* CNN global pooling on either
                // [minibatch, channels, 1, X] or [minibatch, channels, X, 1] data
                // with a mask array of shape [minibatch, X]

                if (maskArray.rank() != 2) {
                    throw new UnsupportedOperationException(
                                    "Only 2d mask arrays are currently supported for masked global reductions "
                                                    + "on CNN data. Got 4d activations array (shape "
                                                    + Arrays.toString(input.shape()) + ") and " + maskArray.rank()
                                                    + "d mask array (shape " + Arrays.toString(maskArray.shape()) + ") "
                                                    + layerId());
                }

                // FIXME: int cast
                int h = (int) input.size(2);
                int w = (int) input.size(3);
                int maskLength = (int) maskArray.size(1);
                if ((h != 1 && w != 1) || (h != maskLength && w != maskLength)) {
                    throw new UnsupportedOperationException(
                                    "Masked global pooling with on CNN data currently only supports data with h=1 or w=1:\n"
                                                    + " input activations must have shape [minibatchSize,channels,height=1,width] or [minibatchSize,channels,height,width=1] with "
                                                    + " mask array of shape [minibatchSize,width] or [minibatchSize,height] respectively.\n"
                                                    + " Got 4d activations array (shape "
                                                    + Arrays.toString(input.shape()) + ") and " + maskArray.rank()
                                                    + "d mask array (shape " + Arrays.toString(maskArray.shape()) + ") "
                                                    + layerId());
                }

                //Valid combinations of global pooling + masking for CNNs:
                //dimensinos [2,3] with or without reduction
                if (DEFAULT_CNN_POOL_DIMS != poolDim && !Arrays.equals(DEFAULT_CNN_POOL_DIMS, poolDim)) {
                    throw new UnsupportedOperationException(
                                    "Masked global pooling with on CNN data currently only supports poolling over dimensions "
                                                    + "[2,3] (i.e., width and height - both required). Got pooling dimensions "
                                                    + Arrays.toString(poolDim) + ") " + layerId());
                }

                boolean maskAlongHeight = (h == maskLength); //At this point: can't confuse w and h, as one has to be 1...

                reduced2d = MaskedReductionUtil.maskedPoolingConvolution(poolingType, input, maskArray, maskAlongHeight,
                                pNorm);
            } else {
                throw new UnsupportedOperationException("Invalid input: is rank " + input.rank() + " " + layerId());
            }
        }

        //TODO optimize without leverage
        if (collapseDimensions) {
            //Standard/common case
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, reduced2d);
        } else {
            val inputShape = input.shape();
            if (input.rank() == 3) {
                return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, reduced2d.reshape(reduced2d.ordering(), inputShape[0], inputShape[1], 1));
            } else {
                return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, reduced2d.reshape(reduced2d.ordering(), inputShape[0], inputShape[1], 1, 1));
            }
        }
    }

    @Override
    public Layer clone() {
        return new GlobalPoolingLayer(conf);
    }

    private INDArray activateHelperFullArray(INDArray inputArray, int[] poolDim) {
        switch (poolingType) {
            case MAX:
                return inputArray.max(poolDim);
            case AVG:
                return inputArray.mean(poolDim);
            case SUM:
                return inputArray.sum(poolDim);
            case PNORM:
                //P norm: https://arxiv.org/pdf/1311.1780.pdf
                //out = (1/N * sum( |in| ^ p) ) ^ (1/p)
                int pnorm = layerConf().getPnorm();

                INDArray abs = Transforms.abs(inputArray, true);
                Transforms.pow(abs, pnorm, false);
                INDArray pNorm = abs.sum(poolDim);

                return Transforms.pow(pNorm, 1.0 / pnorm, false);
            default:
                throw new RuntimeException("Unknown or not supported pooling type: " + poolingType + " " + layerId());
        }
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        if (!collapseDimensions && epsilon.rank() != 2) {
            val origShape = epsilon.shape();
            //Don't collapse dims case: error should be [minibatch, vectorSize, 1] or [minibatch, channels, 1, 1]
            //Reshape it to 2d, to get rid of the 1s
            epsilon = epsilon.reshape(epsilon.ordering(), origShape[0], origShape[1]);
        }

        Gradient retGradient = new DefaultGradient(); //Empty: no params

        int[] poolDim = null;
        if (input.rank() == 3) {
            if (poolingDimensions == null) {
                //Use default pooling dimensions;
                poolDim = DEFAULT_TIMESERIES_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }

        } else if (input.rank() == 4) {
            //CNN activations
            if (poolingDimensions == null) {
                //Use default pooling dimensions;
                poolDim = DEFAULT_CNN_POOL_DIMS;
            } else {
                poolDim = poolingDimensions;
            }
        }

        INDArray epsilonNd;
        if (maskArray == null) {
            //Standard 'full array' global pooling op
            epsilonNd = epsilonHelperFullArray(input, epsilon, poolDim);
        } else {
            if (input.rank() == 3) {
                epsilonNd = MaskedReductionUtil.maskedPoolingEpsilonTimeSeries(poolingType, input, maskArray, epsilon,
                                pNorm);
            } else if (input.rank() == 4) {

                // FIXME: int cast
                int h =  (int) input.size(2);

                boolean maskAlongHeight = (h == maskArray.size(1));
                epsilonNd = MaskedReductionUtil.maskedPoolingEpsilonCnn(poolingType, input, maskArray, epsilon,
                                maskAlongHeight, pNorm);
            } else {
                throw new UnsupportedOperationException(layerId());
            }

        }

        //TODO optimize without leverage
        epsilonNd = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNd);
        return new Pair<>(retGradient, epsilonNd);
    }

    private INDArray epsilonHelperFullArray(INDArray inputArray, INDArray epsilon, int[] poolDim) {

        //Broadcast: occurs on the remaining dimensions, after the pool dimensions have been removed.
        //TODO find a more efficient way to do this
        int[] broadcastDims = new int[inputArray.rank() - poolDim.length];
        int count = 0;
        for (int i = 0; i < inputArray.rank(); i++) {
            if (ArrayUtils.contains(poolDim, i))
                continue;
            broadcastDims[count++] = i;
        }

        switch (poolingType) {
            case MAX:
                INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(inputArray.dup(), poolDim));
                return Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(isMax, epsilon, isMax, broadcastDims));
            case AVG:
                //if out = avg(in,dims) then dL/dIn = 1/N * dL/dOut
                int n = 1;
                for (int d : poolDim) {
                    n *= inputArray.size(d);
                }
                INDArray ret = Nd4j.create(inputArray.shape());
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(ret, epsilon, ret, broadcastDims));
                ret.divi(n);

                return ret;
            case SUM:
                INDArray retSum = Nd4j.create(inputArray.shape());
                Nd4j.getExecutioner().exec(new BroadcastCopyOp(retSum, epsilon, retSum, broadcastDims));
                return retSum;
            case PNORM:
                int pnorm = layerConf().getPnorm();

                //First: do forward pass to get pNorm array
                INDArray abs = Transforms.abs(inputArray, true);
                Transforms.pow(abs, pnorm, false);

                INDArray pNorm = Transforms.pow(abs.sum(poolDim), 1.0 / pnorm);

                //dL/dIn = dL/dOut * dOut/dIn
                //dOut/dIn = in .* |in|^(p-2) /  ||in||_p^(p-1), where ||in||_p is the output p-norm

                INDArray numerator;
                if (pnorm == 2) {
                    numerator = inputArray.dup();
                } else {
                    INDArray absp2 = Transforms.pow(Transforms.abs(inputArray, true), pnorm - 2, false);
                    numerator = inputArray.mul(absp2);
                }

                INDArray denom = Transforms.pow(pNorm, pnorm - 1, false);
                denom.rdivi(epsilon);
                Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(numerator, denom, numerator, broadcastDims));

                return numerator;
            default:
                throw new RuntimeException("Unknown or not supported pooling type: " + poolingType + " " + layerId());
        }
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Global pooling layer: no masking is possible after this point... i.e., masks have been taken into account
        // as part of the pooling
        this.maskArray = maskArray;
        this.maskState = null; //Not used in global pooling - always applied

        return null;
    }
}
