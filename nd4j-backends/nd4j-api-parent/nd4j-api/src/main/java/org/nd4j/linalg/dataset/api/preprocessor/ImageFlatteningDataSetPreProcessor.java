package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * A DataSetPreProcessor used to flatten a 4d CNN features array to a flattened 2d format (for use in networks such
 * as a DenseLayer/multi-layer perceptron)
 *
 * @author Alex Black
 */
public class ImageFlatteningDataSetPreProcessor implements DataSetPreProcessor {
    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray input = toPreProcess.getFeatures();
        if (input.rank() == 2)
            return; //No op: should usually never happen in a properly configured data pipeline

        //Assume input is standard rank 4 activations - i.e., CNN image data
        //First: we require input to be in c order. But c order (as declared in array order) isn't enough; also need strides to be correct
        if (input.ordering() != 'c' || !Shape.strideDescendingCAscendingF(input))
            input = input.dup('c');

        val inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        val outShape = new long[] {inShape[0], inShape[1] * inShape[2] * inShape[3]};

        INDArray reshaped = input.reshape('c', outShape);
        toPreProcess.setFeatures(reshaped);
    }
}
