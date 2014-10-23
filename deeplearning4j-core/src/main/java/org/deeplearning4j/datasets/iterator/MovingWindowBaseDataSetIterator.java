package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.iterator.impl.MovingWindowDataSetFetcher;
import org.nd4j.linalg.dataset.DataSet;

/**
 *
 * DataSetIterator for moving window (rotating matrices)
 *
 * @author Adam Gibson
 */
public class MovingWindowBaseDataSetIterator extends BaseDatasetIterator {
    public MovingWindowBaseDataSetIterator(int batch, int numExamples, DataSet data,int windowRows,int windowColumns) {
        super(batch, numExamples, new MovingWindowDataSetFetcher(data,windowRows,windowColumns));
    }



}
