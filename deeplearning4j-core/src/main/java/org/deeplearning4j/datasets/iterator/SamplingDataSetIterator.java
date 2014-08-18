package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.linalg.dataset.DataSet;

/**
 * A wrapper for a dataset to sample from.
 * This will randomly sample from the given dataset.
 * @author Adam GIbson
 */
public class SamplingDataSetIterator implements DataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2700563801361726914L;
	private DataSet sampleFrom;
	private int batchSize;
	private int totalNumberSamples;
	private int numTimesSampled;


    /**
     *
     * @param sampleFrom the dataset to sample from
     * @param batchSize the batch size to sample
     * @param totalNumberSamples the sample size
     */
 	public SamplingDataSetIterator(DataSet sampleFrom, int batchSize,
			int totalNumberSamples) {
		super();
		this.sampleFrom = sampleFrom;
		this.batchSize = batchSize;
		this.totalNumberSamples = totalNumberSamples;
	}

	@Override
	public boolean hasNext() {
		return numTimesSampled < totalNumberSamples;
	}

	@Override
	public DataSet next() {
		DataSet ret = sampleFrom.sample(batchSize);
		numTimesSampled+= batchSize;
		return ret;
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalExamples() {
		return totalNumberSamples * batchSize;
	}

	@Override
	public int inputColumns() {
		return sampleFrom.numInputs();
	}

	@Override
	public int totalOutcomes() {
		return sampleFrom.numOutcomes();
	}

	@Override
	public void reset() {
		numTimesSampled = 0;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return numTimesSampled;
	}

	@Override
	public int numExamples() {
		return sampleFrom.numExamples();
	}

	@Override
	public DataSet next(int num) {
		DataSet ret = sampleFrom.sample(num);
		numTimesSampled++;
		return ret;
	}
	
	

}
