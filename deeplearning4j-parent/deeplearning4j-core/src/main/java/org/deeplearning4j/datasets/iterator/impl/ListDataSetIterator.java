package org.deeplearning4j.datasets.iterator.impl;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Wraps a data set collection
 * @author Adam Gibson
 *
 */
public class ListDataSetIterator implements DataSetIterator {



	/**
	 * 
	 */
	private static final long serialVersionUID = -7569201667767185411L;
	private int curr = 0;
	private int batch = 10;
	private List<DataSet> list;

	public ListDataSetIterator(Collection<DataSet> coll,int batch) {
		list = new ArrayList<>(coll);
		this.batch = batch;

	}

	/**
	 * Initializes with a batch of 5
	 * @param coll the collection to iterate over
	 */ 
	public ListDataSetIterator(Collection<DataSet> coll) {
		this(coll,5);

	}

	@Override
	public boolean hasNext() {
		return curr < list.size();
	}

	@Override
	public DataSet next() {
		int range = curr + batch;
		if(range >= list.size()) {
			range = list.size();
		}

		DataSet d =  DataSet.merge(list.subList(curr, range));
		curr += batch;
		return d;
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalExamples() {
		return list.size();
	}

	@Override
	public int inputColumns() {
		return list.get(0).getFirst().columns;
	}

	@Override
	public int totalOutcomes() {
		return list.get(0).getSecond().columns;
	}

	@Override
	public void reset() {
		curr = 0;
	}

	@Override
	public int batch() {
		return batch;
	}

	@Override
	public int cursor() {
		return curr;
	}

	@Override
	public int numExamples() {
		return list.size();
	}


}
