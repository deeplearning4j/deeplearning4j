package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;

public class BaseDatasetIterator implements DataSetIterator {

	private int batch;
	private DataSetFetcher fetcher;
	
	
	
	public BaseDatasetIterator(int batch,DataSetFetcher fetcher) {
		this.batch = batch;
		this.fetcher = fetcher;
	}

	@Override
	public boolean hasNext() {
		return fetcher.hasMore();
	}

	@Override
	public Pair<DoubleMatrix, DoubleMatrix> next() {
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
	
	

	

}
