package org.deeplearning4j.datasets.iterator;

/**
 * @author Adam Gibson
 */
public class ReutersNewsGroupsDataSetIterator extends BaseDatasetIterator {
    public ReutersNewsGroupsDataSetIterator(int batch, int numExamples, DataSetFetcher fetcher) {
        super(batch, numExamples, fetcher);
    }
}
