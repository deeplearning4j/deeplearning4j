package org.deeplearning4j.word2vec.vectorizer;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.topicmodeling.CountVectorizer;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.viterbi.Index;
import org.jblas.DoubleMatrix;
/**
 * Bag of words vectorizer.
 * Transforms a document in to a bag of words
 * @author Adam Gibson
 *
 */
public class BagOfWordsVectorizer implements Vectorizer {

	private SentenceIterator sentenceIter;
	private TokenizerFactory tokenizerFactory;
	private int numLabels;
	private int label;
	private Index vocab;
	
	/**
	 * Converts a document in to a bag of words
	 * @param iter the sentence iterator to use
	 * This handles segmenting the document in to 
	 * whole segments
	 * @param tokenizer the tokenizer to use
	 * @param label the label of this document
	 * @param numLabels the total number of labels for the problem
	 * @param index a word vocab
	 */
	public BagOfWordsVectorizer(SentenceIterator iter,TokenizerFactory tokenizer,int label,int numLabels,Index index) {
		this.sentenceIter = iter;
		this.tokenizerFactory = tokenizer;
		this.numLabels = numLabels;
		this.label = label;
		this.vocab = index;
	}

	/**
	 * Creates a bag of words vectorizer with no vocab.
	 * Note that this will count all words in the document.
	 * @param iter the iterator used to segment the document
	 * @param tokenizer the tokenizer to use for word frequencies
	 * @param label the label of this document
	 * @param numLabels the number of labels for the domain
	 */
	public BagOfWordsVectorizer(SentenceIterator iter,TokenizerFactory tokenizer,int label,int numLabels) {
		this.sentenceIter = iter;
		this.tokenizerFactory = tokenizer;
		this.numLabels = numLabels;
		this.label = label;
		this.vocab = new Index();
	}
	
	@Override
	public DataSet vectorize() {
		CountVectorizer count = new CountVectorizer(sentenceIter, tokenizerFactory, vocab);
		DoubleMatrix input = count.toVector();
		DoubleMatrix d = MatrixUtil.toOutcomeVector(label, numLabels);
		return new DataSet(input,d);
	}

}
