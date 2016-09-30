package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 5/25/16.
 * Standard scaler calculates a moving column wise
 * variance and mean
 * http://www.johndcook.com/blog/standard_deviation/
 */
public class NormalizerStandardize implements DataNormalization {
    private static Logger logger = LoggerFactory.getLogger(NormalizerStandardize.class);
    private int runningTotal , labelRunningTotal = 0;
    private int batchCount,labelbatchCount = 0;
    private int featureRank = 2;
    private INDArray featureMeanStd, labelMeanStd;
    private INDArray featureMean, featureStd, labelMean, labelStd;
    private boolean fitLabels = false;

    private INDArray fit(INDArray theArray) {
        INDArray theMean, theStd;
        theMean = theArray.mean(0);
        theStd = theArray.std(0);
        theStd.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
        if (theStd.min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
            logger.info("API_INFO: Std deviation found to be zero. Transform will round upto epsilon to avoid nans.");
        return Nd4j.vstack(theMean,theStd).dup();
    }

    private void runningFit(INDArray thenewArray, INDArray currentMeanStd, int runningTotal, boolean allDone) {
        int batchCount = thenewArray.size(0);
        INDArray currentMean = currentMeanStd.getRow(0);
        INDArray currentStd = currentMeanStd.getRow(1);
        if (!allDone) {
            // m_newM = m_oldM + (x - m_oldM)/m_n;
            INDArray xMinusMean = thenewArray.subRowVector(currentMean);
            INDArray newMean = currentMean.add(xMinusMean.sum(0).divi(runningTotal));
            // Using http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
            // for a version of calc variance when dataset is partitioned into two sample sets
            // Also described in https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            // delta = mean_B - mean_A; A is data seen so far, B is the current batch
            // M2 is the var*n
            // M2 = M2_A + M2_B + delta^2 * nA * nB/(nA+nB)
            INDArray meanB = thenewArray.mean(0);
            INDArray deltaSq = Transforms.pow(meanB.subRowVector(currentMean), 2);
            INDArray deltaSqScaled = deltaSq.mul(((float) runningTotal - batchCount) * batchCount / (float) runningTotal);
            INDArray mtwoB = Transforms.pow(thenewArray.std(0), 2);
            mtwoB.muli(batchCount);
            currentStd.addi(mtwoB);
            currentStd.addi(deltaSqScaled);
            currentMeanStd.putRow(0,newMean);
        }
        else {
            /*
            currentMeanStd.getRow(1).divi(runningTotal);
            currentMeanStd.putRow(1,Transforms.sqrt(currentMeanStd.getRow(1)));
            currentMeanStd.getRow(1).addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
            */
            currentStd.divi(runningTotal);
            Transforms.sqrt(currentStd,false);
            currentStd.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
            if (currentMeanStd.getRow(0).min(1) == Nd4j.scalar(Nd4j.EPS_THRESHOLD))
                logger.info("API_INFO: Std deviation found to be zero. Transform will round upto epsilon to avoid nans.");
        }
    }

    /**
     * Flag to specify if the labels/outputs in the dataset should be also normalized
     * default value is false
     * @param fitLabels
     */

    public void fitLabel(boolean fitLabels) {
        this.fitLabels = fitLabels;
    }

    /**
     * Fit the given model with dataset
     * to calculate mean and std dev with
     * @param dataSet
     */
    public void fit(DataSet dataSet) {
        //int size, sizeExcludeMask;
        featureRank = dataSet.getFeatures().rank();

        INDArray theFeatures = dataSet.getFeatures();
        if (featureRank == 3) theFeatures = DataSetUtil.tailor3d2d(dataSet,true);
        if (featureRank == 4) theFeatures = DataSetUtil.tailor4d2d(dataSet,true);

        featureMeanStd = fit(theFeatures);
        //FIXME:
        /*
        //Entries that are masked are wiped out to zero with tailor3d2d dataset
        //Therefore they don't contribute to the sum etc etc, but the total size needs to be adjusted
        if (dataSet.getFeaturesMaskArray() !=null) {
            size = theFeatures.size(0);
            sizeExcludeMask = dataSet.getFeaturesMaskArray() != null ? dataSet.getFeaturesMaskArray().sumNumber().intValue() : dataSet.getFeatures().size(0);
            featureMeanStd.muli(size).divi(sizeExcludeMask);
        }
        */
        featureMean = featureMeanStd.getRow(0).dup();
        featureStd = featureMeanStd.getRow(1).dup();

        if (fitLabels) {
            INDArray theLabels = dataSet.getLabels();
            if (featureRank == 3) theLabels = DataSetUtil.tailor3d2d(dataSet,false);
            if (featureRank == 4) theLabels = DataSetUtil.tailor4d2d(dataSet,false);
            labelMeanStd = fit(theLabels);
            //FIXME:
            /*
            //Entries that are masked are wiped out to zero with tailor3d2d dataset
            //Therefore they don't contribute to the sum etc etc, but the total size needs to be adjusted
            if (dataSet.getLabelsMaskArray() != null) {
                size = theLabels.size(0);
                sizeExcludeMask = dataSet.getLabelsMaskArray() != null ? dataSet.getLabelsMaskArray().sumNumber().intValue() : dataSet.getFeatures().size(0);
                labelMeanStd.muli(size).divi(sizeExcludeMask);
            }
            */
            labelMean = labelMeanStd.getRow(0).dup();
            labelStd = labelMeanStd.getRow(1).dup();
        }

    }

    /**
     * Fit the given model with a given iterator
     * to calculate mean and std dev with
     * @param iterator
     */
    public void fit(DataSetIterator iterator) {
        featureMeanStd = null;
        batchCount = 0;
        labelbatchCount = 0;
        runningTotal = 0;
        labelRunningTotal = 0;
        INDArray theFeatures, theLabels;
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            theFeatures = next.getFeatures();
            theLabels = next.getLabels();
            if (featureRank == 3) theFeatures = DataSetUtil.tailor3d2d(next,true);
            if (featureRank == 4) theFeatures = DataSetUtil.tailor4d2d(next,true);
            batchCount = theFeatures.size(0);
            runningTotal += batchCount;
            if (fitLabels) {
                if (featureRank == 3) theLabels = DataSetUtil.tailor3d2d(next, false);
                if (featureRank == 4) theLabels = DataSetUtil.tailor4d2d(next, false);
                labelbatchCount = theLabels.size(0);
                labelRunningTotal += labelbatchCount;
            }
            if(featureMeanStd == null) {
                this.fit(next);
            }
            else {
                this.runningFit(theFeatures,featureMeanStd,runningTotal,false);
                if (fitLabels) {
                    this.runningFit(theLabels,labelMeanStd,labelRunningTotal,false);
                }
            }
        }
        if (runningTotal != batchCount) this.runningFit(featureMeanStd,featureMeanStd,runningTotal,true);
        featureMean = featureMeanStd.getRow(0).dup();
        featureStd = featureMeanStd.getRow(1).dup();
        if (fitLabels) {
            if (labelRunningTotal != labelbatchCount) this.runningFit(labelMeanStd,labelMeanStd,labelRunningTotal,true);
            labelMean = labelMeanStd.getRow(0).dup();
            labelStd = labelMeanStd.getRow(1).dup();
        }
        iterator.reset();
    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        if (featureMean == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        INDArray theFeatures = toPreProcess.getFeatures();
        INDArray theLabels = toPreProcess.getLabels();
        this.preProcess(theFeatures,true);
        if (fitLabels) this.preProcess(theLabels,false);
    }

    private void preProcess(INDArray theFeatures, boolean isFeatures) {
        INDArray mean, std;
        mean = isFeatures ? featureMean : labelMean;
        std = isFeatures ? featureStd : labelStd;
        if (featureRank == 2) {
            theFeatures.subiRowVector(mean);
            theFeatures.diviRowVector(std);
        }
        // if feature Rank is 3 (time series) samplesxfeaturesxtimesteps
        // if feature Rank is 4 (images) samplesxchannelsxrowsxcols
        // both cases operations should be carried out in dimension 1
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(theFeatures,mean,theFeatures,1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(theFeatures,std,theFeatures,1));
        }

    }

    /**
     * Transform the given dataset
     * @param toPreProcess
     */
    @Override
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    /**
     * Transform the given INDArray
     * @param theFeatures
     */
    @Override
    public void transform(INDArray theFeatures) {
        this.transform(theFeatures,true);
    }

    public void transform(INDArray theArray, boolean isFeatures) {
        this.preProcess(theArray,isFeatures);
    }

    /**
     * Revert the data to what it was before transform
     * @param toPreProcess the dataset to revert back
     */
    public void revert(DataSet toPreProcess) {
        if (featureMean== null || featureStd == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        if (featureRank == 2) {
            toPreProcess.getFeatures().muliRowVector(featureStd);
            toPreProcess.getFeatures().addiRowVector(featureMean);
        }
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(toPreProcess.getFeatures(),featureStd,toPreProcess.getFeatures(),1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(toPreProcess.getFeatures(),featureMean,toPreProcess.getFeatures(),1));
        }
    }

    public INDArray getMean() {
        if (featureMean == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return featureMean;
    }

    public INDArray getLabelMean() {
        if (featureMean == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return labelMean;
    }

    public INDArray getStd() {
        if (featureStd == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return featureStd;
    }

    public INDArray getLabelStd() {
        if (featureStd == null) throw new RuntimeException("API_USE_ERROR: Preprocessors have to be explicitly fit before use. Usage: .fit(dataset) or .fit(datasetiterator)");
        return labelStd;
    }

    /**
     * Load the given mean and std
     *@param statistics the statistics to laod
     * @throws IOException
     */
    @Override
    public void load(File...statistics) throws IOException {
        this.featureMean = Nd4j.readBinary(statistics[0]);
        this.featureStd = Nd4j.readBinary(statistics[1]);
        if (fitLabels) {
            this.labelMean = Nd4j.readBinary(statistics[2]);
            this.labelStd = Nd4j.readBinary(statistics[3]);
        }
    }

    /**
     * Save the current mean and std
     * @param statistics the statistics to save
     * @throws IOException
     */
    @Override
    public void save(File...statistics) throws IOException {
        Nd4j.saveBinary(this.featureMean,statistics[0]);
        Nd4j.saveBinary(this.featureStd,statistics[1]);
        if (fitLabels) {
            Nd4j.saveBinary(this.labelMean,statistics[2]);
            Nd4j.saveBinary(this.labelStd,statistics[3]);
        }
    }

}
