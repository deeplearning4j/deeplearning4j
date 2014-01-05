package com.ccc.deeplearning.word2vec;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import scala.concurrent.Future;
import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.berkeley.MapFactory;
import com.ccc.deeplearning.berkeley.Triple;
import com.ccc.deeplearning.util.MatrixUtil;
import com.ccc.deeplearning.word2vec.viterbi.Index;

/**
 * Leveraging a 3 layer neural net with a softmax approach as output,
 * converts a word based on its context and the training examples in to a
 * numeric vector
 * @author Adam Gibson
 *
 */
public class Word2Vec implements Serializable {


	private static final long serialVersionUID = -2367495638286018038L;
	private Map<String,VocabWord> vocab = new HashMap<String,VocabWord>();
	/* all words; including those not in the actual ending index */
	private Map<String,VocabWord> rawVocab = new HashMap<String,VocabWord>();

	private Map<Integer,String> indexToWord = new HashMap<Integer,String>();
	private Random rand = new Random(1);
	private int topNSize = 40;
	public int EXP_TABLE_SIZE = 500;
	//matrix row of a given word
	private Index wordIndex = new Index();
	/* pre calculated sigmoid table */
	private double[] expTable = new double[EXP_TABLE_SIZE];
	private int sample = 1;
	//learning rate
	private Double alpha = 0.025;
	private int MAX_EXP = 6;
	private int wordCount  = 0;
	public final static double MIN_ALPHA =  0.001;
	//number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
	private int minWordFrequency = 5;
	//context to use for gathering word frequencies
	private int window = 5;
	private int trainWordsCount = 0;
	//number of neurons per layer
	private int layerSize = 50;
	private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
	private int size = 0;
	private int words = 0;
	//input layer
	private DoubleMatrix syn0;
	//hidden layer
	private DoubleMatrix syn1;
	private List<String> sentences = new ArrayList<String>();
	private int allWordsCount = 0;
	private static ActorSystem trainingSystem;
	private List<String> stopWords;
	/* out of vocab */
	private double[] oob;
	/*
	 * Used as a pair for when
	 * the number of sentences is not known
	 */
	private Iterator<File> fileIterator;
	private int numLines;

	/**
	 * Mainly meant for use with
	 * static loading methods.
	 * Please consider one of the other constructors
	 * That being {@link #Word2Vec(Iterator)} (streaming dataset)
	 * or         {@link #Word2Vec(Collection)}
	 * or         {@link #Word2Vec(Collection, int)}
	 */
	public Word2Vec() {
		createExpTable();
		oob = new double[layerSize];
		Arrays.fill(oob,1.0);
		readStopWords();
	}

	private void readStopWords() {
		try {
			stopWords = IOUtils.readLines(new ClassPathResource("/stopwords").getInputStream());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	/**
	 * This is meant for streaming a dataset
	 * alongside with add {@link #addToVocab(String)}
	 * @param fileIterator the iterator over the dataset
	 */
	public Word2Vec(Iterator<File> fileIterator) {
		this();
		this.fileIterator = fileIterator;
		readStopWords();

	}

	/**
	 * Assumes whole dataset is passed in. 
	 * This is purely meant for batch methods.
	 * Same as calling {@link #Word2Vec(Collection, int)}
	 * with the second argument being 5
	 * @param sentences the sentences to use
	 * to train on
	 */
	public Word2Vec(Collection<String> sentences) {
		this(sentences,5);
		readStopWords();

	}

	/**
	 * Initializes based on assumption of whole data set being passed in.
	 * @param sentences the sentences to be used for training
	 * @param minWordFrequency the minimum word frequency
	 * to be counted in the vocab
	 */
	public Word2Vec(Collection<String> sentences,int minWordFrequency) {
		createExpTable();
		this.minWordFrequency = minWordFrequency;
		this.sentences = new ArrayList<String>(sentences);
		this.buildVocab(sentences);

		oob = new double[layerSize];
		Arrays.fill(oob,1.0);
		readStopWords();

	}


	public double[] getWordVector(String word) {
		int i = this.wordIndex.indexOf(word);
		if(i < 0)
			return oob;
		return syn0.getRow(i).toArray();
	}

	public DoubleMatrix getWordVectorMatrix(String word) {
		int i = this.wordIndex.indexOf(word);
		if(i < 0)
			return new DoubleMatrix(oob);
		return syn0.getRow(i);
	}


	public VocabWord getWord(String key) {
		return vocab.get(key);
	}


	/**
	 * Precompute the exp() table
	 * f(x) = x / (x + 1)
	 */
	private void createExpTable() {
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
	}






	private void insertTopN(String name, double score, List<VocabWord> wordsEntrys) {
		if (wordsEntrys.size() < topNSize) {
			wordsEntrys.add(new VocabWord(score,layerSize));
			return;
		}
		double min = Float.MAX_VALUE;
		int minOffe = 0;
		for (int i = 0; i < topNSize; i++) {
			VocabWord wordEntry = wordsEntrys.get(i);
			if (min > wordEntry.getWordFrequency()) {
				min = (double) wordEntry.getWordFrequency();
				minOffe = i;
			}
		}

		if (score > min) {
			wordsEntrys.set(minOffe, new VocabWord(score,layerSize));
		}

	}

	public boolean hasWord(String word) {
		return wordIndex.indexOf(word) >= 0;
	}

	public void train() {
		if(trainingSystem == null)
			trainingSystem = ActorSystem.create();

		if(stopWords == null)
			readStopWords();
		log.info("Training word2vec multithreaded");

		MapFactory<String,Double> factory = new MapFactory<String,Double>() {

			private static final long serialVersionUID = 5447027920163740307L;

			@Override
			public Map<String, Double> buildMap() {
				return new java.util.concurrent.ConcurrentHashMap<String,Double>();
			}

		};

		final Counter<String> totalWords = new Counter<String>(factory);


		//came in through empty constructor, build vocab.
		//the other scenario is being initialized through a file iterator
		//this means that vocab should have already been trained
		//but not the vectors
		if(syn0 == null && !sentences.isEmpty()) {
			this.buildVocab(sentences);
		}


		if(syn0.rows != this.vocab.size())
			throw new IllegalStateException("We appear to be missing vectors here. Unable to train. Please ensure vectors were loaded properly.");

		if(sentences.isEmpty()) {
			//no sentences or file iterator defined
			if(fileIterator == null)
				throw new IllegalStateException("Unable to train sentences, no iterator or sentences defined");

			else {

				if(!fileIterator.hasNext()) 
					throw new IllegalStateException("File iterator does not appear to have any files to train on");
				/*
				 * Count the number of lines
				 */
				final CountDownLatch sentenceCounter = new CountDownLatch(numLines);
				int numLinesIterated = 0;
				while(fileIterator.hasNext()) {
					try {
						LineIterator lines = FileUtils.lineIterator(fileIterator.next());
						while(lines.hasNext()) {
							final String sentence = lines.nextLine();
							numLinesIterated++;
							processSentence(sentence, sentenceCounter,totalWords);

						}


					} catch (Exception e) {
						throw new RuntimeException(e);
					}

				}

				try {
					sentenceCounter.await();
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
				}

				if(numLinesIterated != numLines) {
					this.numLines = numLinesIterated;
				}
			}
		}


		else {

			final CountDownLatch sentenceCounter = new CountDownLatch(sentences.size());


			for(final String sentence : sentences) 
				processSentence(sentence, sentenceCounter,totalWords);


			try {
				sentenceCounter.await();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}
		}


		if(trainingSystem != null)
			trainingSystem.shutdown();

	}

	private void processSentence(final String sentence,final CountDownLatch sentenceCounter,final Counter<String> totalWords) {
		Future<Void> future = Futures.future(new Callable<Void>() {

			@Override
			public Void call() throws Exception {
				if(!sentence.isEmpty())
					trainSentence(sentence, totalWords);

				return null;
			}

		},trainingSystem.dispatcher());


		future.onComplete(new OnComplete<Void>() {

			@Override
			public void onComplete(Throwable arg0, Void arg1)
					throws Throwable {

				sentenceCounter.countDown();

				if(sentenceCounter.getCount() % 10000 == 0) {
					alpha = new Double(Math.max(MIN_ALPHA, alpha * (1 - 1.0 * totalWords.totalCount() / allWordsCount)));
					log.info("Alpha updated " + alpha + " progress " + sentenceCounter.getCount() + " sentence size " + sentences.size());
				}

			}

		},trainingSystem.dispatcher());


	}


	public List<VocabWord> trainSentence(String sentence,Counter<String> totalWords) {
		StringTokenizer tokenizer = new StringTokenizer(sentence);
		List<VocabWord> sentence2 = new ArrayList<VocabWord>();
		while(tokenizer.hasMoreTokens()) {
			String next = tokenizer.nextToken();
			VocabWord word = vocab.get(next);
			if(word == null) 
				continue;

			sentence2.add(word);
			totalWords.incrementCount(next, 1.0);

		}
		trainSentence(sentence2);
		return sentence2;
	}





	/**
	 * The format for saving the word2vec model is as follows.
	 * 
	 * @param file
	 * @throws IOException
	 */
	public void saveModel(File file) throws IOException {

		if(file.exists())
			file.delete();

		try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
				new FileOutputStream(file)))) {
			dataOutputStream.writeInt(vocab.size());
			dataOutputStream.writeInt(layerSize);

			for(int i = 0; i < vocab.size(); i++) {
				String word = this.wordIndex.get(i).toString();
				dataOutputStream.writeUTF(word);
				vocab.get(word).write(dataOutputStream);

			}

			syn0.out(dataOutputStream);
			syn1.out(dataOutputStream);



			dataOutputStream.flush();
			dataOutputStream.close();

		} catch (IOException e) {
			log.error("Unable to save model",e);
		}

	}

	public void loadModel(InputStream is) throws Exception {
		try(DataInputStream dis = new DataInputStream(new BufferedInputStream(is))) {
			int vocabSize = dis.readInt();
			int layerSize = dis.readInt();
			setLayerSize(layerSize);
			for(int i = 0; i < vocabSize; i++) {
				String word = dis.readUTF();
				wordIndex.add(word);
				vocab.put(word,new VocabWord().read(dis, layerSize));

			}

			syn0.in(dis);
			syn1.in(dis);

			dis.close();

		}
		catch(IOException e) {
			log.error("Unable to read file for loading model",e);
		}
	}









	/**
	 *
	 * 
	 * @param word
	 * @return
	 */
	public Set<VocabWord> distance(String word) {
		DoubleMatrix wordVector = getWordVectorMatrix(word);
		if (wordVector == null) {
			return null;
		}
		DoubleMatrix tempVector = null;
		List<VocabWord> wordEntrys = new ArrayList<VocabWord>(topNSize);
		String name = null;
		for (int i = 0; i < syn0.rows; i++) {
			name = wordIndex.get(i).toString();
			if (name.equals(word)) {
				continue;
			}
			double dist = 0;
			tempVector = syn0.getRow(i);
			dist = wordVector.dot(tempVector);
			insertTopN(name, dist, wordEntrys);
		}
		return new TreeSet<VocabWord>(wordEntrys);
	}

	/**
	 *
	 * @return 
	 */
	public TreeSet<VocabWord> analogy(String word0, String word1, String word2) {
		DoubleMatrix wv0 = getWordVectorMatrix(word0);
		DoubleMatrix wv1 = getWordVectorMatrix(word1);
		DoubleMatrix wv2 = getWordVectorMatrix(word2);


		DoubleMatrix wordVector = wv1.sub(wv1).add(wv2);

		if (wv1 == null || wv2 == null || wv0 == null) 
			return null;

		DoubleMatrix tempVector;
		String name;
		List<VocabWord> wordEntrys = new ArrayList<VocabWord>(topNSize);
		for (int i = 0; i < syn0.rows; i++) {
			name = wordIndex.get(i).toString();

			if (name.equals(word0) || name.equals(word1) || name.equals(word2)) {
				continue;
			}
			tempVector = syn0.getRow(i);
			double dist = wordVector.dot(tempVector);
			insertTopN(name, dist, wordEntrys);
		}
		return new TreeSet<VocabWord>(wordEntrys);
	}

	/**
	 * Meant for streaming methods of 
	 * adding to the vocabulary.
	 * House keeping related to the 
	 * file iterator that will be needed
	 * when training the word vectors later on
	 * (assumed to be passed in)
	 * is also done within this method.
	 * 
	 * Note that an IllegalStateException is also
	 * thrown when sentences is not empty and this method is called.
	 * This ensures a consistent state.
	 * @param words the words to be added
	 */
	public void addToVocab(String words) {
		if(!sentences.isEmpty())
			throw new IllegalStateException("Only one method (complete sentences passed in) or streaming is allowed. Please clear sentences and pass in a file iterator to use the other method");
		int count = 0;
		//
		numLines++;
		StringTokenizer tokenizer = new StringTokenizer(words);

		this.allWordsCount += tokenizer.countTokens();
		count++;
		if(count % 10000 == 0)
			log.info("Processed  sentence " + count + " current word count " + allWordsCount);

		while(tokenizer.hasMoreTokens()) {
			String token = tokenizer.nextToken();
			VocabWord word = rawVocab.get(token);

			//this will also increment the
			//vocab word at the final level
			//due to the reference being the same
			if(word != null)
				word.increment();
			else {
				word = new VocabWord(1.0,layerSize);
				rawVocab.put(token,word);
			}


			if(word.getWordFrequency() >= minWordFrequency) {
				word.setIndex(wordIndex.size());
				wordIndex.add(token);
				this.vocab.put(token, word);
			}

		}

	}

	public void setup() {

		log.info("Building binary tree");
		buildBinaryTree();
		log.info("Resetting weights");
		resetWeights();
	}


	public void buildVocab(Collection<String> sentences) {
		readStopWords();
		Queue<String> queue = new ArrayDeque<>(sentences);
		int count = 0;
		while(!queue.isEmpty()) {
			final String words = queue.poll();

			StringTokenizer tokenizer = new StringTokenizer(words);

			this.allWordsCount += tokenizer.countTokens();
			count++;
			if(count % 10000 == 0)
				log.info("Processed  sentence " + count + " current word count " + allWordsCount);

			while(tokenizer.hasMoreTokens()) {
				String token = tokenizer.nextToken();
				VocabWord word = rawVocab.get(token);

				//this will also increment the
				//vocab word at the final level
				//due to the reference being the same
				if(word != null)
					word.increment();
				else {
					word = new VocabWord(1.0,layerSize);
					rawVocab.put(token,word);
				}
				//note that for purposes of word frequency, the 
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(word.getWordFrequency() >= minWordFrequency && !stopWords.contains(token)) {
					if(!this.vocab.containsKey(token)) {
						word.setIndex(this.vocab.size());
						this.vocab.put(token, word);
						wordIndex.add(token);

					}


				}

			}

		}
		setup();

	}

	public void addSentence(String sentence) {
		this.sentences.add(sentence);
	}

	public void trainSentence(List<VocabWord> sentence) {
		long nextRandom = 5;
		for(int i = 0; i < sentence.size(); i++) {
			VocabWord entry = sentence.get(i);
			// The subsampling randomly discards frequent words while keeping the ranking same
			if (sample > 0) {
				double ran = (Math.sqrt(entry.getWordFrequency() / (sample * trainWordsCount)) + 1)
						* (sample * trainWordsCount) / entry.getWordFrequency();
				nextRandom = nextRandom * 25214903917L + 11;
				if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
					continue;
				}
			}
			nextRandom = nextRandom * 25214903917L + 11;
			int b = (int) nextRandom % window;
			skipGram(i,sentence,b);
		}
	}



	public void skipGram(int i,List<VocabWord> sentence,int b) {
		VocabWord word = sentence.get(i);
		if(word == null)
			return;


		//subsampling
		for(int j = b; j < window * 2 + 1 - b; j++) {
			if(j == window)
				continue;
			int c1 = i - window + j;

			if (c1 < 0 || c1 >= sentence.size()) 
				continue;

			VocabWord word2 = sentence.get(c1);
			iterate(word,word2);
		}
	}

	public void  iterate(VocabWord w1,VocabWord w2) {
		DoubleMatrix l1 = syn0.getRow(w2.getIndex());
		DoubleMatrix l2a = syn1.getRows(w1.getCodes());
		DoubleMatrix fa = MatrixUtil.sigmoid(MatrixUtil.dot(l1, l2a.transpose()));
		// ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
		DoubleMatrix ga = DoubleMatrix.ones(fa.length).sub(MatrixUtil.toMatrix(w1.getCodes())).sub(fa).mul(alpha);
		DoubleMatrix outer = ga.mmul(l1);
		for(int i = 0; i < w1.getPoints().length; i++) {
			DoubleMatrix toAdd = l2a.getRow(i).add(outer.getRow(i));
			syn1.putRow(w1.getPoints()[i],toAdd);
		}

		DoubleMatrix updatedInput = l1.add(MatrixUtil.dot(ga, l2a));
		syn0.putRow(w2.getIndex(),updatedInput);
	}

	public DoubleMatrix toExp(DoubleMatrix input) {
		for(int i = 0; i < input.length; i++) {
			double f = input.get(i);
			f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
			f = expTable[(int) f];
			input.put(i,f);
		}
		return input;
	}


	/* Builds the binary tree for the word relationships */
	private void buildBinaryTree() {
		PriorityQueue<VocabWord> heap = new PriorityQueue<VocabWord>(vocab.values());
		int i = 0;
		while(heap.size() > 1) {
			VocabWord min1 = heap.poll();
			VocabWord min2 = heap.poll();


			VocabWord add = new VocabWord(min1.getWordFrequency() + min2.getWordFrequency(),layerSize);
			int index = (vocab.size() + i);

			add.setIndex(index); 
			add.setLeft(min1);
			add.setRight(min2);
			min1.setCode(0);
			min2.setCode(1);
			min1.setParent(add);
			min2.setParent(add);
			heap.add(add);
			i++;
		}

		Triple<VocabWord,int[],int[]> triple = new Triple<VocabWord,int[],int[]>(heap.poll(),new int[]{},new int[]{});
		Stack<Triple<VocabWord,int[],int[]>> stack = new Stack<>();
		stack.add(triple);
		while(!stack.isEmpty())  {
			triple = stack.pop();
			int[] codes = triple.getSecond();
			int[] points = triple.getThird();
			VocabWord node = triple.getFirst();
			if(node == null) {
				log.info("Node was null");
				continue;
			}
			if(node.getIndex() < vocab.size()) {
				node.setCodes(codes);
				node.setPoints(points);
			}
			else {
				int[] copy = plus(points,node.getIndex() - vocab.size());
				points = copy;
				triple.setThird(points);
				stack.add(new Triple<VocabWord,int[],int[]>(node.getLeft(),plus(codes,0),points));
				stack.add(new Triple<VocabWord,int[],int[]>(node.getRight(),plus(codes,1),points));

			}
		}



		log.info("Built tree");
	}

	private int[] plus (int[] addTo,int add) {
		int[] copy = new int[addTo.length + 1];
		for(int c = 0; c < addTo.length; c++)
			copy[c] = addTo[c];
		copy[addTo.length] = add;
		return copy;
	}

	/* reinit weights */
	private void resetWeights() {
		syn1 = DoubleMatrix.zeros(vocab.size(), layerSize);
		//self.syn0 += (random.rand(len(self.vocab), self.layer1_size) - 0.5) / self.layer1_size
		syn0 = DoubleMatrix.zeros(vocab.size(),layerSize);
		org.jblas.util.Random.seed(1);
		for(int i = 0; i < syn0.rows; i++)
			for(int j = 0; j < syn0.columns; j++) {
				syn0.put(i,j,(org.jblas.util.Random.nextDouble() - 0.5) / layerSize);
			}
	}


	/**
	 * Returns the similarity of 2 words
	 * @param word the first word
	 * @param word2 the second word
	 * @return a normalized similarity (cosine similarity)
	 */
	public double similarity(String word,String word2) {
		if(word.equals(word2))
			return 1.0;

		double[] vector = getWordVector(word);
		double[] vector2 = getWordVector(word2);
		if(vector == null || vector2 == null)
			return -1;
		DoubleMatrix d1 = MatrixUtil.unitVec(new DoubleMatrix(vector));
		DoubleMatrix d2 = MatrixUtil.unitVec(new DoubleMatrix(vector2));
		return d1.dot(d2);

	}


	public void setMinWordFrequency(int minWordFrequency) {
		this.minWordFrequency = minWordFrequency;
	}
	public int getMAX_EXP() {
		return MAX_EXP;
	}




	public int getLayerSize() {
		return layerSize;
	}
	public void setLayerSize(int layerSize) {
		this.layerSize = layerSize;
	}
	public void setMAX_EXP(int mAX_EXP) {
		MAX_EXP = mAX_EXP;
	}


	public int getEXP_TABLE_SIZE() {
		return EXP_TABLE_SIZE;
	}


	public void setEXP_TABLE_SIZE(int eXP_TABLE_SIZE) {
		EXP_TABLE_SIZE = eXP_TABLE_SIZE;
	}


	public double[] getExpTable() {
		return expTable;
	}


	public void setExpTable(double[] expTable) {
		this.expTable = expTable;
	}


	public int getTrainWordsCount() {
		return trainWordsCount;
	}


	public Index getWordIndex() {
		return wordIndex;
	}
	public void setWordIndex(Index wordIndex) {
		this.wordIndex = wordIndex;
	}


	public DoubleMatrix getSyn0() {
		return syn0;
	}

	public DoubleMatrix getSyn1() {
		return syn1;
	}


	public Map<String, VocabWord> getVocab() {
		return vocab;
	}




	public Map<Integer, String> getIndexToWord() {
		return indexToWord;
	}

	public Random getRand() {
		return rand;
	}




	public double getAlpha() {
		return alpha;
	}




	public int getWordCount() {
		return wordCount;
	}




	public int getMinWordFrequency() {
		return minWordFrequency;
	}




	public int getWindow() {
		return window;
	}

	public Map<String, VocabWord> getRawVocab() {
		return rawVocab;
	}

	public int getTopNSize() {
		return topNSize;
	}

	public int getSample() {
		return sample;
	}



	public int getSize() {
		return size;
	}

	public double[]	 getOob() {
		return oob;
	}

	public int getWords() {
		return words;
	}

	public List<String> getSentences() {
		return sentences;
	}

	public int getAllWordsCount() {
		return allWordsCount;
	}

	public static ActorSystem getTrainingSystem() {
		return trainingSystem;
	}

	public Iterator<File> getFileIterator() {
		return fileIterator;
	}

	public void setSyn0(DoubleMatrix syn0) {
		this.syn0 = syn0;
	}

	public void setSyn1(DoubleMatrix syn1) {
		this.syn1 = syn1;
	}


}
