package org.nd4j.linalg.dataset.api.iterator;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Standard scaler calculates a moving column wise
 * variance and mean
 * http://www.johndcook.com/blog/standard_deviation/
 */
public class StandardScaler {
    private INDArray mean,std;
    private int runningTotal = 0;


    public void fit(DataSet dataSet) {
        mean = dataSet.getFeatureMatrix().mean(0);
        std = dataSet.getFeatureMatrix().std(0);
    }

    /**
     * Fit the given model
     * @param iterator the data to iterate oer
     */
    public void fit(DataSetIterator iterator) {
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            runningTotal += next.numExamples();
            if(mean == null) {
                //start with the mean and std of zero
                //column wise
                mean = next.getFeatureMatrix().mean(0);
                std = (iterator.batch() == 1) ? Nd4j.zeros(mean.shape()) : Transforms.pow(next.getFeatureMatrix().std(0),2);
                std.muli(iterator.batch());
            }
            else {
                // m_newM = m_oldM + (x - m_oldM)/m_n;
                // This only works if batch size is 1, m_newS = m_oldS + (x - m_oldM)*(x - m_newM);
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
                INDArray deltaSqScaled = deltaSq.mul(((float)runningTotal-iterator.batch())*iterator.batch()/iterator.totalExamples());
                INDArray mtwoB = Transforms.pow(next.getFeatureMatrix().std(0),2);
                mtwoB.muli(iterator.batch());
                std = std.add(mtwoB);
                std = std.add(deltaSqScaled);
                mean = newMean;
            }

        }
        std.divi(runningTotal);
        std = Transforms.sqrt(std);
        iterator.reset();
    }


    /**
     * Load the given mean and std
     * @param mean the mean file
     * @param std the std file
     * @throws IOException
     */
    public void load(File mean,File std) throws IOException {
        this.mean = Nd4j.readBinary(mean);
        this.std = Nd4j.readBinary(std);
    }

    /**
     * Save the current mean and std
     * @param mean the mean
     * @param std the std
     * @throws IOException
     */
    public void save(File mean,File std) throws IOException {
        Nd4j.saveBinary(this.mean,mean);
        Nd4j.saveBinary(this.std,std);
    }

    /**
     * Transform the data
     * @param dataSet the dataset to transform
     */
    public void transform(DataSet dataSet) {
        dataSet.setFeatures(dataSet.getFeatures().subiRowVector(mean));
        dataSet.setFeatures(dataSet.getFeatures().diviRowVector(std));
    }


    public INDArray getMean() {
        return mean;
    }

    public INDArray getStd() {
        return std;
    }
}
