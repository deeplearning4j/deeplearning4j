package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Standard scaler calculates a moving column wise
 * variance and mean
 * http://www.johndcook.com/blog/standard_deviation/
 */
public class NormalizerStandardize implements DataNormalization {
    private static Logger logger = LoggerFactory.getLogger(NormalizerStandardize.class);
    private INDArray mean,std;
    private int runningTotal = 0;
    private int batchCount = 0;

    /**
     * Fit the given model with dataset
     * to calculate mean and std dev with
     * @param dataSet
     */
    public void fit(DataSet dataSet) {
        int featureRank = dataSet.getFeatureMatrix().rank();
        INDArray theFeatures = dataSet.getFeatureMatrix();
        // If 3d or 4d dataset convert to 2d
        if (featureRank > 2) {
            if (featureRank == 3) theFeatures = tailor3d2d(dataSet);
            if (featureRank == 4) theFeatures = tailor4d2d(dataSet);
        }

        mean = theFeatures.mean(0);
        std = theFeatures.std(0);
        std.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (std.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
            logger.info("API_INFO: Std deviation found to be zero. Transform will round upto epsilon to avoid nans.");
    }

    /**
     * Fit the given model with a given iterator
     * to calculate mean and std dev with
     * @param iterator
     */
    public void fit(DataSetIterator iterator) {
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            INDArray theFeatures = next.getFeatureMatrix();
            int featureRank = theFeatures.rank();
            if (featureRank > 2) {
                if (featureRank == 3) theFeatures = tailor3d2d(next);
                if (featureRank == 4) theFeatures = tailor4d2d(next);
            }
            batchCount = next.getFeaturesMaskArray() != null ? next.getFeaturesMaskArray().sumNumber().intValue() :  theFeatures.size(0);
            runningTotal += batchCount;
            if(mean == null) {
                mean = theFeatures.mean(0);
                std = std.muli(batchCount);
            }
            else {
                // m_newM = m_oldM + (x - m_oldM)/m_n;
                INDArray xMinusMean = theFeatures.subRowVector(mean);
                INDArray newMean = mean.add(xMinusMean.sum(0).divi(runningTotal));
                // Using http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
                // for a version of calc variance when dataset is partitioned into two sample sets
                // Also described in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
                // delta = mean_B - mean_A; A is data seen so far, B is the current batch
                // M2 is the var*n
                // M2 = M2_A + M2_B + delta^2 * nA * nB/(nA+nB)
                INDArray meanB = theFeatures.mean(0);
                INDArray deltaSq = Transforms.pow(meanB.subRowVector(mean),2);
                INDArray deltaSqScaled = deltaSq.mul(((float)runningTotal-batchCount)*batchCount/(float)runningTotal);
                INDArray mtwoB = Transforms.pow(theFeatures.std(0),2);
                mtwoB.muli(batchCount);
                std = std.add(mtwoB);
                std = std.add(deltaSqScaled);
                mean = newMean;
            }
        }
        std.divi(runningTotal);
        std = Transforms.sqrt(std);
        std.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (std.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
            logger.info("API_INFO: Std deviation found to be zero. Transform will round upto epsilon to avoid nans.");
        iterator.reset();
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        if (mean == null || std == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        toPreProcess.getFeatures().subiRowVector(mean);
        toPreProcess.getFeatures().diviRowVector(std);
    }

    /**
     * Transform the given dataset
     * @param toPreProcess
     */
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    /**
     * Transform the dataset from given iterator
     * Need not set preprocessor on the iterator in this case
     * @param toPreProcessIter the dataset to transform
     */
    public void transform(DataSetIterator toPreProcessIter) {
        while (toPreProcessIter.hasNext()) {
            this.preProcess(toPreProcessIter.next());
        }
        toPreProcessIter.reset();
    }



    public void revertPreProcess(DataSet toPreProcess) {
        if (mean == null || std == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        toPreProcess.getFeatures().muliRowVector(std);
        toPreProcess.getFeatures().addiRowVector(mean);
    }

    /**
     * Revert the data to what it was before transform
     * @param toPreProcess the dataset to revert back
     */
    public void revert(DataSet toPreProcess) {
        if (toPreProcess.getFeatureMatrix().rank() == 2)
            this.revertPreProcess(toPreProcess);
        else
            throw new RuntimeException("API_USE_ERROR: Reverting not supported for feature matrices with rank larger than 2");

    }

    public void revert(DataSetIterator toPreProcessIter) {
        while (toPreProcessIter.hasNext()) {
            this.revertPreProcess(toPreProcessIter.next());
        }
        toPreProcessIter.reset();
    }

    public INDArray getMean() {
        if (mean == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return mean;
    }

    public INDArray getStd() {
        if (std == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return std;
    }

    /**
     * Load the given mean and std
     *@param statistics the statistics to laod
     * @throws IOException
     */
    @Override
    public void load(File...statistics) throws IOException {
        this.mean = Nd4j.readBinary(statistics[0]);
        this.std = Nd4j.readBinary(statistics[1]);
    }

    /**
     * Save the current mean and std
     * @param statistics the statistics to save
     * @throws IOException
     */
    @Override
    public void save(File...statistics) throws IOException {
        Nd4j.saveBinary(this.mean,statistics[0]);
        Nd4j.saveBinary(this.std,statistics[1]);
    }

    private INDArray tailor3d2d(DataSet dataset) {
        /* A 2d dataset has dimemsions sample x features
         * A 3d dataset is a timeseries with dimensions sample x features x timesteps
         * A 3d dataset can also have a mask associated with it in case samples are of varying time steps
         * Each sample has a mask associated with it that is applied to all features.
         * Masks are of dimension sample x timesteps
         */
        int instances = dataset.getFeatureMatrix().size(0);
        int features = dataset.getFeatureMatrix().size(1);
        int timesteps = dataset.getFeatureMatrix().size(2);

        boolean hasMasks = dataset.getFeaturesMaskArray() != null;
        INDArray in2d = Nd4j.create(features,timesteps*instances);

        int tads = dataset.getFeatureMatrix().tensorssAlongDimension(2,0);
        // the number of tads are the number of features
        for(int i = 0; i < tads; i++){
            INDArray thisTAD = dataset.getFeatureMatrix().tensorAlongDimension(i, 2, 0);
            //mask is samples x timesteps
            if (hasMasks)
                //if there are masks they are multiplied with the mask array to wipe out the values associated with it
                //to wipe out the values associated with it to wipe out the values associated with it
                thisTAD.muli(dataset.getFeaturesMaskArray());
            //Each row is now values for a given feature across all time steps, across all samples
            in2d.putRow(i, Nd4j.toFlattened('c',thisTAD));
        }
        //Must transpose to return a matrix compatible with 2d viz samples x features
        in2d = in2d.transpose();
        //flatten mask
        if (hasMasks) {
            //only need rows where columnMask is 1
            INDArray columnMask = Nd4j.toFlattened('c',dataset.getFeaturesMaskArray()).transpose();
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

    private INDArray tailor4d2d(DataSet dataset) {
        int instances = dataset.getFeatureMatrix().size(0);
        int channels = dataset.getFeatureMatrix().size(1);
        int height = dataset.getFeatureMatrix().size(2);
        int width = dataset.getFeatureMatrix().size(2);

        INDArray in2d = Nd4j.create(channels,height*width*instances);

        int tads = dataset.getFeatureMatrix().tensorssAlongDimension(3,2,0);
        for(int i = 0; i < tads; i++){
            INDArray thisTAD = dataset.getFeatureMatrix().tensorAlongDimension(i, 2, 0);
            in2d.putRow(i, Nd4j.toFlattened(thisTAD));
        }
        return in2d.transposei();
    }

}
