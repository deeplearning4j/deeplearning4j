package com.ccc.deeplearning.datasets.iterator;

import com.ccc.deeplearning.datasets.DataSet;
/**
 * Baseline implementation includes
 * control over the data fetcher and some basic
 * getters for metadata
 * @author Adam Gibson
 *
 */
public class BaseDatasetIterator implements DataSetIterator {


	private static final long serialVersionUID = -116636792426198949L;
	private int batch,numExamples;
	private DataSetFetcher fetcher;
	
	
	
	public BaseDatasetIterator(int batch,int numExamples,DataSetFetcher fetcher) {
		this.batch = batch;
		if(numExamples < 0)
			numExamples = fetcher.totalExamples();
		
		this.numExamples = numExamples;
		this.fetcher = fetcher;
	}

	@Override
	public boolean hasNext() {
		return fetcher.hasMore() && fetcher.cursor() < numExamples;
	}

	@Override
	public DataSet next() {
		fetcher.fetch(batch);
		return fetcher.next();
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalExamples() {
		return fetcher.totalExamples();
	}

	@Override
	public int inputColumns() {
		return fetcher.inputColumns();
	}

	@Override
	public int totalOutcomes() {
		return fetcher.totalOutcomes();
	}

	@Override
	public void reset() {
		fetcher.reset();
	}

	@Override
	public int batch() {
		return batch;
	}

	@Override
	public int cursor() {
		return fetcher.cursor();
	}

	@Override
	public int numExamples() {
		return numExamples;
	}
	
	

	

}
