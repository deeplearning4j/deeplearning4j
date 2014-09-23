package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.util.Index;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * In memory lookup cache for smaller datasets
 *
 * @author Adam Gibson
 */
public class InMemoryLookupCache implements VocabCache,Serializable {

	private Index wordIndex = new Index();
	private boolean useAdaGrad = true;
	private Counter<String> wordFrequencies = Util.parallelCounter();
	private Map<String,VocabWord> vocabs = new ConcurrentHashMap<>();
	private Map<Integer,INDArray> codes = new ConcurrentHashMap<>();
	private INDArray syn0,syn1;
	private int vectorLength = 50;
	private RandomGenerator rng = new MersenneTwister(123);
	private AtomicInteger totalWordOccurrences = new AtomicInteger(0);
	private float lr = 1e-1f;


	public InMemoryLookupCache(int vectorLength) {
		this(vectorLength,true);

	}

	/**
	 * Initialization constructor for pre loaded models
	 * @param vectorLength the vector length
	 * @param vocabSize the vocab  size
	 */
	public InMemoryLookupCache(int vectorLength,int vocabSize) {
		this.vectorLength = vectorLength;
		syn0 = Nd4j.rand(vocabSize,vectorLength);
	}


	public InMemoryLookupCache(int vectorLength,boolean useAdaGrad) {
		this.vectorLength = vectorLength;
		this.useAdaGrad = useAdaGrad;

	}

	public InMemoryLookupCache(int vectorLength,boolean useAdaGrad,float lr) {
		this.vectorLength = vectorLength;
		this.useAdaGrad = useAdaGrad;
		this.lr = lr;

	}


	/**
	 * Iterate on the given 2 vocab words
	 *
	 * @param w1 the first word to iterate on
	 * @param w2 the second word to iterate on
	 */
	@Override
	public  void iterate(VocabWord w1, VocabWord w2) {
		if(w1.getCodes() == null)
			return;
		if(w2.getIndex() >= syn0.rows())
			return;

		int l1 = w2.getIndex();
		//current word vector
		INDArray syn0 = this.syn0.getRow(l1);

		//error for current word and context
		INDArray neu1e = Nd4j.create(vectorLength);


		float avgChange = 0.0f;
		for(int d = 0; d < w1.getCodes().length; d++) {
			int inner = w1.getPoints()[d];
			int l2 = inner;
			//other word vector
			INDArray syn1 = this.syn1.getRow(l2);
			if(l1 >= this.syn0.rows() || l2 >= this.syn1.rows())
				break;

			float dot = Nd4j.getBlasWrapper().dot(syn0,syn1);

			//score
			float f = (float) MathUtils.sigmoid(dot);
			//gradient
			float g = (1 - w1.getCodes()[d] - f);
			float lr = useAdaGrad ? w1.getLearningRate(d,g) : this.lr;

			g *= lr;
			avgChange += g;
			Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
			Nd4j.getBlasWrapper().axpy(g, syn0, syn1);

		}

		avgChange /=  w1.getCodes().length;
		if(useAdaGrad)
			Nd4j.getBlasWrapper().axpy(avgChange,neu1e,syn0);
		else
			Nd4j.getBlasWrapper().axpy(1,neu1e,syn0);

	}

	/**
	 * Returns all of the words in the vocab
	 *
	 * @returns all the words in the vocab
	 */
	@Override
	public  synchronized Collection<String> words() {
		return vocabs.keySet();
	}

	/**
	 * Reset the weights of the cache
	 */
	@Override
	public void resetWeights() {
		int words = vocabs.size();

		syn0  = Nd4j.randn(new int[]{words,vectorLength}).subi(0.5f).divi(vectorLength);
		syn1 = Nd4j.create(syn0.shape());

	}

	/**
	 * Increment the count for the given word
	 *
	 * @param word the word to increment the count for
	 */
	@Override
	public synchronized void incrementWordCount(String word) {
		incrementWordCount(word,1);
	}

	/**
	 * Increment the count for the given word by
	 * the amount increment
	 *
	 * @param word      the word to increment the count for
	 * @param increment the amount to increment by
	 */
	@Override
	public  synchronized void incrementWordCount(String word, int increment) {
		wordFrequencies.incrementCount(word,1);
		if(containsWord(word)) {
			VocabWord word2 = wordFor(word);
			word2.increment(increment);


		}
		totalWordOccurrences.set(totalWordOccurrences.get() + increment);

	}

	/**
	 * Returns the number of times the word has occurred
	 *
	 * @param word the word to retrieve the occurrence frequency for
	 * @return 0 if hasn't occurred or the number of times
	 * the word occurs
	 */
	@Override
	public int wordFrequency(String word) {
		return (int) wordFrequencies.getCount(word);
	}

	/**
	 * Returns true if the cache contains the given word
	 *
	 * @param word the word to check for
	 * @return
	 */
	@Override
	public boolean containsWord(String word) {
		return vocabs.containsKey(word);
	}

	/**
	 * Returns the word contained at the given index or null
	 *
	 * @param index the index of the word to get
	 * @return the word at the given index
	 */
	@Override
	public String wordAtIndex(int index) {
		return (String) wordIndex.get(index);
	}

	/**
	 * Returns the index of a given word
	 *
	 * @param word the index of a given word
	 * @return the index of a given word or -1
	 * if not found
	 */
	@Override
	public int indexOf(String word) {
		return wordIndex.indexOf(word);
	}

	/**
	 * @param codeIndex
	 * @param code
	 */
	@Override
	public void putCode(int codeIndex, INDArray code) {
		codes.put(codeIndex,code);
	}

	/**
	 * Loads the co-occurrences for the given codes
	 *
	 * @param codes the codes to load
	 * @return an ndarray of code.length by layerSize
	 */
	@Override
	public INDArray loadCodes(int[] codes) {
		return syn1.getRows(codes);
	}

	/**
	 * Returns all of the vocab word nodes
	 *
	 * @return
	 */
	@Override
	public Collection<VocabWord> vocabWords() {
		return vocabs.values();
	}

	/**
	 * The total number of word occurrences
	 *
	 * @return the total number of word occurrences
	 */
	@Override
	public int totalWordOccurrences() {
		return  totalWordOccurrences.get();
	}

	/**
	 * Inserts a word vector
	 *
	 * @param word   the word to insert
	 * @param vector the vector to insert
	 */
	@Override
	public void putVector(String word, INDArray vector) {
		if(word == null)
			throw new IllegalArgumentException("No null words allowed");
		if(vector == null)
			throw new IllegalArgumentException("No null vectors allowed");
		int idx = indexOf(word);
		syn0.putRow(idx,vector);

	}

	/**
	 * @param word
	 * @return
	 */
	@Override
	public INDArray vector(String word) {
		if(word == null)
			return null;
		return syn0.getRow(indexOf(word));
	}

	/**
	 * @param word
	 * @return
	 */
	@Override
	public VocabWord wordFor(String word) {
		return vocabs.get(word);
	}

	/**
	 * @param index
	 * @param word
	 */
	@Override
	public synchronized void addWordToIndex(int index, String word) {
		if(!wordFrequencies.containsKey(word))
			wordFrequencies.incrementCount(word,1);
		wordIndex.add(word,index);
	}

	/**
	 * @param word
	 * @param vocabWord
	 */
	@Override
	public synchronized void putVocabWord(String word, VocabWord vocabWord) {
		addWordToIndex(vocabWord.getIndex(),word);
		vocabs.put(word,vocabWord);
		wordIndex.add(word,vocabWord.getIndex());

	}

	/**
	 * Returns the number of words in the cache
	 *
	 * @return the number of words in the cache
	 */
	@Override
	public synchronized int numWords() {
		return vocabs.size();
	}
}
