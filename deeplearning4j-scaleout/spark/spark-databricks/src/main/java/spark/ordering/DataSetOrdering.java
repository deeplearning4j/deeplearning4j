package spark.ordering;

import org.nd4j.linalg.dataset.DataSet;
import scala.Function1;
import scala.Some;
import scala.math.Ordering;

/**
 * Orders by data set size.
 * This will force the dataset with a certain number of mini batches to be grouped at th end.
 */
public class DataSetOrdering implements Ordering<DataSet> {
    @Override
    public Some<Object> tryCompare(DataSet dataSet, DataSet t1) {
        return null;
    }

    @Override
    public int compare(DataSet dataSet, DataSet t1) {
        return 0;
    }

    @Override
    public boolean lteq(DataSet dataSet, DataSet t1) {
        return dataSet.numExamples() >= t1.numExamples();
    }

    @Override
    public boolean gteq(DataSet dataSet, DataSet t1) {
        return !lteq(dataSet, t1);
    }

    @Override
    public boolean lt(DataSet dataSet, DataSet t1) {
        return dataSet.numExamples() >= t1.numExamples();
    }

    @Override
    public boolean gt(DataSet dataSet, DataSet t1) {
        return !lt(dataSet, t1);
    }

    @Override
    public boolean equiv(DataSet dataSet, DataSet t1) {
        return dataSet.numExamples() == t1.numExamples();
    }

    @Override
    public DataSet max(DataSet dataSet, DataSet t1) {
       return gt(dataSet, t1) ? dataSet : t1;
     }

    @Override
    public DataSet min(DataSet dataSet, DataSet t1) {
        return max(dataSet, t1) == dataSet ? t1 : dataSet;
     }

    @Override
    public Ordering<DataSet> reverse() {
        return null;
    }

    @Override
    public <U> Ordering<U> on(Function1<U, DataSet> function1) {
        return null;
    }

    @Override
    public Ops mkOrderingOps(DataSet dataSet) {
        return null;
    }
}
