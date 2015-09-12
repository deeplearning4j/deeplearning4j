package org.nd4j.linalg.dataset.api.iterator;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Standard scaler calculates a moving column wise
 * variance and mean
 * http://www.johndcook.com/blog/standard_deviation/
 */
public class StandardScaler {
    private INDArray mean,std;
    private int runningTotal = 0;

    /**
     * Fit the given model
     * @param iterator the data to iterate oer
     */
    public void fit(DataSetIterator iterator) {
        while(iterator.hasNext()) {
            DataSet next = iterator.next();
            if(mean == null) {
                //start with the mean and std of zero
                //column wise
                mean = next.getFeatureMatrix().mean(0);
                std = Nd4j.zeros(mean.shape());
            }
            else {
                // m_newM = m_oldM + (x - m_oldM)/m_n;
                // m_newS = m_oldS + (x - m_oldM)*(x - m_newM);
                INDArray xMinusMean = next.getFeatureMatrix().subRowVector(mean);
                INDArray newMean = mean.add(xMinusMean.sum(0).divi(runningTotal));
                std.addi(xMinusMean.muli(next.getFeatureMatrix().subRowVector(newMean)).sum(0).divi(runningTotal));
                mean = newMean;
            }

            runningTotal += next.numExamples();

        }

        iterator.reset();




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
