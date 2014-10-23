package org.deeplearning4j.datasets.iterator.impl;

import java.io.IOException;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * Mnist data applyTransformToDestination iterator.
 * @author Adam Gibson
 */
public class MnistDataSetIterator extends BaseDatasetIterator {

	public MnistDataSetIterator(int batch,int numExamples) throws IOException {
		this(batch,numExamples,true);
	}

    /**
     * Whether to binarize the data or not
     * @param batch the the batch size of the examples
     * @param numExamples the overall number of examples
     * @param binarize whether to binarize mnist or not
     * @throws IOException
     */
    public MnistDataSetIterator(int batch,int numExamples,boolean binarize) throws IOException {
        super(batch, numExamples,new MnistDataFetcher(binarize));
    }


}
