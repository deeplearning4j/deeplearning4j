package org.nd4j.linalg.dataset.api;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Created by susaneraly on 9/20/16.
 */
public class DataSetUtil {
    public static INDArray tailor2d(@NonNull DataSet dataSet, boolean areFeatures) {
        return tailor2d(areFeatures ? dataSet.getFeatures() : dataSet.getLabels(),
                        areFeatures ? dataSet.getFeaturesMaskArray() : dataSet.getLabelsMaskArray());
    }

    public static INDArray tailor2d(@NonNull INDArray data, INDArray mask) {
        switch (data.rank()) {
            case 1:
            case 2:
                return data;
            case 3:
                return tailor3d2d(data, mask);
            case 4:
                return tailor4d2d(data);
            default:
                throw new RuntimeException("Unsupported data rank");
        }
    }

    /**
     * @deprecated
     */
    public static INDArray tailor3d2d(DataSet dataset, boolean areFeatures) {
        INDArray data = areFeatures ? dataset.getFeatures() : dataset.getLabels();
        INDArray mask = areFeatures ? dataset.getFeaturesMaskArray() : dataset.getLabelsMaskArray();
        return tailor3d2d(data, mask);
    }

    public static INDArray tailor3d2d(@NonNull INDArray data, INDArray mask) {
        //Check mask shapes:
        if (mask != null) {
            if (data.size(0) != mask.size(0) || data.size(2) != mask.size(1)) {
                throw new IllegalArgumentException(
                                "Invalid mask array/data combination: got data with shape [minibatch, vectorSize, timeSeriesLength] = "
                                                + Arrays.toString(data.shape())
                                                + "; got mask with shape [minibatch,timeSeriesLength] = "
                                                + Arrays.toString(mask.shape())
                                                + "; minibatch and timeSeriesLength dimensions must match");
            }
        }


        if (data.ordering() != 'f' || data.isView() || !Shape.strideDescendingCAscendingF(data)) {
            data = data.dup('f');
        }
        //F order: strides are like [1, miniBatch, minibatch*size] - i.e., each time step array is contiguous in memory
        //This can be reshaped to 2d with a no-copy op
        //Same approach as RnnToFeedForwardPreProcessor in DL4J
        //I.e., we're effectively stacking time steps for all examples

        int[] shape = data.shape();
        INDArray as2d;
        if (shape[0] == 1) {
            as2d = data.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        } else if (shape[2] == 1) {
            as2d = data.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            INDArray permuted = data.permute(0, 2, 1); //Permute, so we get correct order after reshaping
            as2d = permuted.reshape('f', shape[0] * shape[2], shape[1]);
        }

        if (mask == null) {
            return as2d;
        }

        //With stride 1 along the examples (dimension 0), we are concatenating time series - same as the
        if (mask.ordering() != 'f' || mask.isView() || !Shape.strideDescendingCAscendingF(mask)) {
            mask = mask.dup('f');
        }

        INDArray mask1d = mask.reshape('f', new int[] {mask.length(), 1});

        //Assume masks are 0s and 1s: then sum == number of elements
        int numElements = mask.sumNumber().intValue();
        if (numElements == mask.length()) {
            return as2d; //All are 1s
        }
        if (numElements == 0) {
            return null;
        }

        int[] rowsToPull = new int[numElements];
        float[] floatMask1d = mask1d.data().asFloat();
        int currCount = 0;
        for (int i = 0; i < floatMask1d.length; i++) {
            if (floatMask1d[i] != 0.0f) {
                rowsToPull[currCount++] = i;
            }
        }

        INDArray subset = Nd4j.pullRows(as2d, 1, rowsToPull); //Tensor along dimension 1 == rows
        return subset;
    }

    public static INDArray tailor4d2d(DataSet dataset, boolean areFeatures) {
        return tailor4d2d(areFeatures ? dataset.getFeatures() : dataset.getLabels());
    }

    public static INDArray tailor4d2d(@NonNull INDArray data) {
        int instances = data.size(0);
        int channels = data.size(1);
        int height = data.size(2);
        int width = data.size(3);

        INDArray in2d = Nd4j.create(channels, height * width * instances);

        int tads = data.tensorssAlongDimension(3, 2, 0);
        for (int i = 0; i < tads; i++) {
            INDArray thisTAD = data.tensorAlongDimension(i, 3, 2, 0);
            in2d.putRow(i, Nd4j.toFlattened(thisTAD));
        }
        return in2d.transposei();
    }

    public static void setMaskedValuesToZero(INDArray data, INDArray mask) {
        if (mask == null || data.rank() != 3)
            return;

        Nd4j.getExecutioner().exec(new BroadcastMulOp(data, mask, data, 0, 2));
    }
}
