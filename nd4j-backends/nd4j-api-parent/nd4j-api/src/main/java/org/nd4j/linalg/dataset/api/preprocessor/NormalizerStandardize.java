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
        if (!dataSet.hasMaskArrays()) {
            mean = dataSet.getFeatureMatrix().mean(0);
            std = dataSet.getFeatureMatrix().std(0);
            std.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
            if (std.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
                logger.info("API_INFO: Std deviation found to be zero. Transform will round upto epsilon to avoid nans.");
        }
        else {
            mean = dataSet.getFeatureMatrix().mulRowVector(dataSet.getFeaturesMaskArray()).mean(0);
            std = dataSet.getFeatureMatrix().std(0);
            std.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
            if (std.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
                logger.info("API_INFO: Std deviation found to be zero. Transform will round upto epsilon to avoid nans.");

        }
    }

    /**
     * Fit the given model with a given iterator
     * to calculate mean and std dev with
     * @param iterator
     */
    public void fit(DataSetIterator iterator) {
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            runningTotal += next.numExamples();
            batchCount = next.getFeatures().size(0);
            if(mean == null) {
                //start with the mean and std of zero
                //column wise
                mean = next.getFeatureMatrix().mean(0);
                std = (batchCount == 1) ? Nd4j.zeros(mean.shape()) : Transforms.pow(next.getFeatureMatrix().std(0),2);
                std.muli(batchCount);
            }
            else {
                // m_newM = m_oldM + (x - m_oldM)/m_n;
                INDArray xMinusMean = next.getFeatureMatrix().subRowVector(mean);
                INDArray newMean = mean.add(xMinusMean.sum(0).divi(runningTotal));
                // Using http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
                // for a version of calc variance when dataset is partitioned into two sample sets
                // Also described in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
                // delta = mean_B - mean_A; A is data seen so far, B is the current batch
                // M2 is the var*n
                // M2 = M2_A + M2_B + delta^2 * nA * nB/(nA+nB)
                INDArray meanB = next.getFeatureMatrix().mean(0);
                INDArray deltaSq = Transforms.pow(meanB.subRowVector(mean),2);
                INDArray deltaSqScaled = deltaSq.mul(((float)runningTotal-batchCount)*batchCount/(float)runningTotal);
                INDArray mtwoB = Transforms.pow(next.getFeatureMatrix().std(0),2);
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
    public void revert(DataSet toPreProcess) {this.revertPreProcess(toPreProcess);}

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
}
