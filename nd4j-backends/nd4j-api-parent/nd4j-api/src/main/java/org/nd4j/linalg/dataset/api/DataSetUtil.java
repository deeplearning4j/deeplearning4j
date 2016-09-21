package org.nd4j.linalg.dataset.api;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 9/20/16.
 */
public class DataSetUtil {

    public static INDArray tailor3d2d(DataSet dataset, boolean areFeatures) {
        /* A 2d dataset has dimemsions sample x features
         * A 3d dataset is a timeseries with dimensions sample x features x timesteps
         * A 3d dataset can also have a mask associated with it in case samples are of varying time steps
         * Each sample has a mask associated with it that is applied to all features.
         * Masks are of dimension sample x timesteps
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
        //flatten mask
        if (hasMasks) {
            //only need rows where columnMask is 1
            INDArray columnMask = Nd4j.toFlattened('c',theMask).transpose();
            int actualSamples = columnMask.sumNumber().intValue();
            INDArray in2dMask = Nd4j.create(actualSamples,features);
            int j = 0;
            for (int i=0; i < timesteps*instances; i++){
                if (columnMask.getInt(i, 0) != 0) {
                    in2dMask.putRow(j, in2d.getRow(i));
                    j++;
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
