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

    /**
     *
     * @param batch the mini batch size
     * @param numExamples the number of examples
     * @param is the input stream to read from
     * @param labelColumn the index (0 based) of the label
     */
    public CSVDataSetIterator(int batch, int numExamples,InputStream is,int labelColumn) {
        super(batch, numExamples, new CSVDataFetcher(is,labelColumn,0));
    }

    /**
     *
     * @param batch the mini batch size
     * @param numExamples the number of examples
     * @param f the file to read from
     * @param labelColumn the index (0 based) of the label
     * @throws IOException
     */
    public CSVDataSetIterator(int batch, int numExamples,File f,int labelColumn) throws IOException {
        super(batch, numExamples, new CSVDataFetcher(f,labelColumn,0));
    }

    /**
     *
     * @param batch the mini batch size
     * @param numExamples the number of examples
     * @param is the input stream to read from
     * @param labelColumn the index (0 based) of the label
     * @param skipLines the number of lines to skip
     */
    public CSVDataSetIterator(int batch, int numExamples,InputStream is,int labelColumn,int skipLines) {
        super(batch, numExamples, new CSVDataFetcher(is,labelColumn,skipLines));
    }

    /**
     *
     * @param batch the mini batch size
     * @param numExamples the number of examples
     * @param f the file to read from
     * @param labelColumn the index (0 based) of the label
     * @param skipLines the number of lines to skip
     * @throws IOException
     */
    public CSVDataSetIterator(int batch, int numExamples,File f,int labelColumn,int skipLines) throws IOException {
        super(batch, numExamples, new CSVDataFetcher(f,labelColumn,skipLines));
    }
}
