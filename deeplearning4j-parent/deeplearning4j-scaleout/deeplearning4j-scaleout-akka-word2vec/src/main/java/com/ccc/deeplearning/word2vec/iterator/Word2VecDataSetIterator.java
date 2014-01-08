package com.ccc.deeplearning.word2vec.iterator;

import java.util.List;

import com.ccc.deeplearning.datasets.iterator.BaseDatasetIterator;
import com.ccc.deeplearning.word2vec.Word2Vec;

/**
 * Trains a word2vec model and iterates over a file dataset
 * containing text files with sentences such as:
 * W1 W2 <LABEL> SOME POSITIVE EXAMPLE </LABEL> W3 W4 W5
 * @author Adam Gibson
 *
 */
public class Word2VecDataSetIterator extends BaseDatasetIterator {
	@Override
	public boolean hasNext() {
		return fetcher.hasMore();
	}


	private static final long serialVersionUID = 2397051312760991798L;

	
	
	public Word2VecDataSetIterator(int batch, int numExamples,String path,int  minWordFrequency) {
		super(batch, numExamples, new Word2VecDataFetcher(path,minWordFrequency));
	}
	
	
	public Word2VecDataSetIterator(int batch, int numExamples,String path,Word2Vec vec,List<String> labels) {
		super(batch, numExamples, new Word2VecDataFetcher(path,vec,labels));
	}
	
	
	

}
