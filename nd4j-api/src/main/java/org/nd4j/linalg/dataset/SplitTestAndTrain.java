package org.nd4j.linalg.dataset;

/**
 * Split test and train
 *
 * @author Adam Gibson
 */
public class SplitTestAndTrain {

    private DataSet train,test;

    public SplitTestAndTrain(DataSet train, DataSet test) {

        this.train = train;
        this.test = test;
    }

    public DataSet getTest() {
        return test;
    }

    public void setTest(DataSet test) {
        this.test = test;
    }

    public DataSet getTrain() {

        return train;
    }

    public void setTrain(DataSet train) {
        this.train = train;
    }
}
