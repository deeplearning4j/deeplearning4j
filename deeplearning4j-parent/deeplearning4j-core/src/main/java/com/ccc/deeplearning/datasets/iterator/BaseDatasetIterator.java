package com.ccc.deeplearning.datasets.iterator;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;

public class BaseDatasetIterator implements DataSetIterator {

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
