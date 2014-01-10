package com.ccc.deeplearning.word2vec.dataset;

import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Pair;

public class Word2VecDataSet extends Pair<List<String>,DoubleMatrix> {

	public Word2VecDataSet(List<String> first, DoubleMatrix second) {
		super(first, second);
	}


}
