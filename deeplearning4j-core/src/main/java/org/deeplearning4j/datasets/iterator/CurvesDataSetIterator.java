package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.fetchers.CurvesDataFetcher;

import java.io.IOException;

/**
 * Curves data set iterator
 *
 * @author Adam Gibson
 */
public class CurvesDataSetIterator extends BaseDatasetIterator {
    public CurvesDataSetIterator(int batch, int numExamples) throws IOException {
        super(batch, numExamples, new CurvesDataFetcher());
    }
}
