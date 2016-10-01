package org.nd4j.linalg.dataset.api;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 9/20/16.
 */
public class DataSetUtil {

    public static INDArray tailor3d2d(DataSet dataset, boolean areFeatures) {
        /*
         * A 2d dataset has dimemsions sample x features
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
        INDArray theArray, theMask;
        theArray = areFeatures ? dataset.getFeatures() : dataset.getLabels();
        theMask = areFeatures ? dataset.getFeaturesMaskArray() : dataset.getLabelsMaskArray();

        int instances = theArray.size(0);
        int features = theArray.size(1);
        int timesteps = theArray.size(2);

        boolean hasMasks = theMask != null;
        INDArray in2d = Nd4j.create(features,timesteps*instances);

        int tads = theArray.tensorssAlongDimension(2,0);
        // the number of tads are the number of features
        for(int i = 0; i < tads; i++){
            INDArray thisTAD = theArray.tensorAlongDimension(i, 2, 0);
            //mask is samples x timesteps
            if (hasMasks)
                //if there are masks they are multiplied with the mask array to wipe out the values associated with it
                //to wipe out the values associated with it to wipe out the values associated with it
                thisTAD.muli(theMask);
            //Each row is now values for a given feature across all time steps, across all samples
            in2d.putRow(i, Nd4j.toFlattened('c',thisTAD));
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
            INDArray columnMask = Nd4j.toFlattened('c',theMask).transpose();
            int actualSamples = columnMask.sumNumber().intValue();
            INDArray in2dMask = Nd4j.create(actualSamples,features);
            int i = 0;
            //definitely a faster way to do this as you can skip the rest of the timesteps after the first zero in the mask
            //use a boolean mask??
            for (int j=0; j < instances; j++){
                for (int k=0; k < timesteps; k++){
                        if (columnMask.getInt(j*timesteps+k, 0) != 0) {
                            in2dMask.putRow(i, in2d.getRow(j * timesteps + k));
                            i++;
                        }
                        else {
                            continue;
                        }
                }
            }
            return in2dMask;
        }
        return in2d;
    }

    public static INDArray tailor4d2d(DataSet dataset, boolean areFeatures) {
        INDArray theArray;
        theArray = areFeatures ? dataset.getFeatures() : dataset.getLabels();
        int instances = theArray.size(0);
        int channels = theArray.size(1);
        int height = theArray.size(2);
        int width = theArray.size(3);

        INDArray in2d = Nd4j.create(channels,height*width*instances);

        int tads = theArray.tensorssAlongDimension(3,2,0);
        for(int i = 0; i < tads; i++){
            INDArray thisTAD = theArray.tensorAlongDimension(i, 3, 2, 0);
            in2d.putRow(i, Nd4j.toFlattened(thisTAD));
        }
        return in2d.transposei();
    }

}
