package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.fetchers.CSVDataFetcher;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * CSVDataSetIterator
 * CSV reader for a dataset file
 * @author Adam Gibson
 */
public class CSVDataSetIterator extends BaseDatasetIterator {
    public CSVDataSetIterator(int batch, int numExamples,InputStream is,int labelColumn) {
        super(batch, numExamples, new CSVDataFetcher(is,labelColumn));
    }

    public CSVDataSetIterator(int batch, int numExamples,File f,int labelColumn) throws IOException {
        super(batch, numExamples, new CSVDataFetcher(f,labelColumn));
    }
}
