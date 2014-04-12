package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.loader.ReutersNewsGroupsLoader;

/**
 * @author Adam Gibson
 */
public class ReutersNewsGroupsDataSetIterator extends BaseDatasetIterator {
    public ReutersNewsGroupsDataSetIterator(int batch, int numExamples,boolean tfidf) throws Exception {
        super(batch, numExamples, new ReutersNewsGroupsLoader(tfidf));
    }
}
