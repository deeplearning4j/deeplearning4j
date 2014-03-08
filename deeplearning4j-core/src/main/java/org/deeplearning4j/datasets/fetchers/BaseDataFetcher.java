package org.deeplearning4j.datasets.fetchers;

import java.util.List;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetFetcher;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A base class for assisting with creation of matrices
 * with the data set fetcher
 * @author Adam Gibson
 *
 */
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
	
	/**
	 * Creates a feature vector
	 * @param numRows the number of examples
 	 * @return a feature vector
	 */
	protected DoubleMatrix createInputMatrix(int numRows) {
		return new DoubleMatrix(numRows,inputColumns);
	}
	
	/**
	 * Creates an output label matrix
	 * @param outcomeLabel the outcome label to use
	 * @return a binary vector where 1 is set to the
	 * index specified by outcomeLabel
	 */
	protected DoubleMatrix createOutputVector(int outcomeLabel) {
		return MatrixUtil.toOutcomeVector(outcomeLabel, numOutcomes);
	}
	
	protected DoubleMatrix createOutputMatrix(int numRows) {
		return new DoubleMatrix(numRows,numOutcomes);
	}
	
	/**
	 * Initializes this data set fetcher from the passed in datasets
	 * @param examples the examples to use
	 */
	protected void initializeCurrFromList(List<DataSet> examples) {
		
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
