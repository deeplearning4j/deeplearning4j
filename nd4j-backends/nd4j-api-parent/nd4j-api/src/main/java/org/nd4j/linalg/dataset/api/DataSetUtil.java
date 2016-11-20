package org.nd4j.linalg.dataset.api;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 9/20/16.
 */
public class DataSetUtil {
    public static INDArray tailor2d(@NonNull DataSet dataSet, boolean areFeatures) {
        return tailor2d(
            areFeatures ? dataSet.getFeatures() : dataSet.getLabels(),
            areFeatures ? dataSet.getFeaturesMaskArray() : dataSet.getLabelsMaskArray()
        );
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
        /*
         * A 2d dataset has dimensions sample x features
         * Processing a 3d dataset poses additional complexities
         * A 3d dataset is a timeseries with dimensions sample x features x timesteps
         * A 3d dataset can also have a mask associated with it in which case samples are of varying time steps
         *      (
         *        Each sample has a mask associated with it. The length of the mask is the number of time steps in the longest sample.
         *        Therefore the mask array is of dimension sample x timesteps.
         *        The same mask is applied to all features of a given sample.
         *      )
         * tailor3d2d takes in a 3d dataset with masks and returns a 2d array of size:
         *      along zero dimension: number_of_samples, dataset.getFeatures().size(0)  x number of time steps in longest sample, dataset.getFeatures.size(2)
         *      along first dimension: number of features, dataset.getFeatures().size(1)
         * Values which should be masked are wiped out and replaced by zeros
         */
        int instances = data.size(0);
        int features = data.size(1);
        int timesteps = data.size(2);

        boolean hasMasks = mask != null;
        INDArray in2d = Nd4j.create(features, timesteps * instances);

        int tads = data.tensorssAlongDimension(2, 0);
        // the number of tads are the number of features
        for (int i = 0; i < tads; i++) {
            INDArray thisTAD = data.tensorAlongDimension(i, 2, 0);
            //mask is samples x timesteps
            if (hasMasks)
                //if there are masks they are multiplied with the mask array to wipe out the values associated with it
                thisTAD.muli(mask);
            //Each row is now values for a given feature across all time steps, across all samples
            in2d.putRow(i, Nd4j.toFlattened('c', thisTAD));
        }
        //Must transpose to return a matrix compatible with 2d viz samples x features
        in2d = in2d.transpose();
        /*
        The mask doesn't just exist in the context of differing time steps
        But they can also be used to ignore certain output features.
        */
        //flatten mask
        if (hasMasks) {
            /*
                now each row is a single timestep for some sample
             */
            INDArray columnMask = Nd4j.toFlattened('c', mask).transpose();
            int actualSamples = columnMask.sumNumber().intValue();
            INDArray in2dMask = Nd4j.create(actualSamples, features);
            int i = 0;
            //definitely a faster way to do this as you can skip the rest of the timesteps after the first zero in the mask
            //use a boolean mask??
            for (int j = 0; j < instances; j++) {
                for (int k = 0; k < timesteps; k++) {
                    if (columnMask.getInt(j * timesteps + k, 0) != 0) {
                        in2dMask.putRow(i, in2d.getRow(j * timesteps + k));
                        i++;
                    } else {
                        continue;
                    }
                }
            }
            return in2dMask;
        }
        return in2d;
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
}
