package com.ccc.deeplearning.datasets.fetchers;

import java.util.List;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.DataSetFetcher;
import com.ccc.deeplearning.util.MatrixUtil;

public abstract class BaseDataFetcher implements DataSetFetcher {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -859588773699432365L;
	protected int cursor = 0;
	protected int numOutcomes = -1;
	protected int inputColumns = -1;
	protected DataSet curr;
	protected int totalExamples;
	protected static Logger log = LoggerFactory.getLogger(BaseDataFetcher.class);
	
	
	protected DoubleMatrix createInputMatrix(int numRows) {
		return new DoubleMatrix(numRows,inputColumns);
	}
	
	protected DoubleMatrix createOutputVector(int outcomeLabel) {
		return MatrixUtil.toOutcomeVector(outcomeLabel, numOutcomes);
	}
	
	protected DoubleMatrix createOutputMatrix(int numRows) {
		return new DoubleMatrix(numRows,numOutcomes);
	}
	
	protected void initializeCurrFromList(List<Pair<DoubleMatrix,DoubleMatrix>> examples) {
		
		if(examples.isEmpty())
			log.warn("Warning: empty dataset from the fetcher");
		
		DoubleMatrix inputs = createInputMatrix(examples.size());
		DoubleMatrix labels = createOutputMatrix(examples.size());
		for(int i = 0; i < examples.size(); i++) {
			inputs.putRow(i, examples.get(i).getFirst());
			labels.putRow(i,examples.get(i).getSecond());
		}
		curr = new DataSet(inputs,labels);

	}
	
	@Override
	public boolean hasMore() {
		return cursor < totalExamples;
	}

	@Override
	public DataSet next() {
		return curr;
	}

	@Override
	public int totalOutcomes() {
		return numOutcomes;
	}

	@Override
	public int inputColumns() {
		return inputColumns;
	}

	@Override
	public int totalExamples() {
		return totalExamples;
	}

	@Override
	public void reset() {
		cursor = 0;
	}

	@Override
	public int cursor() {
		return cursor;
	}
	
	

	
}
