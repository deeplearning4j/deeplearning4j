package com.ccc.deeplearning.word2vec;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
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
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
	private Map<Integer,String> indexToWord = new HashMap<Integer,String>();
	private Random rand = new Random(1);
	private int topNSize = 40;
	public int EXP_TABLE_SIZE = 1000;
	//matrix row of a given word
	private Index wordIndex = new Index();

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
	private int layerSize = 100;
	private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
	private int size = 0;
	private int words = 0;
	private DoubleMatrix syn0;
	private DoubleMatrix syn1;
	private List<String> sentences;
	private int allWordsCount = 0;
	private static ActorSystem trainingSystem;
	private double[] oob;

	public Word2Vec() {
		createExpTable();
		oob = new double[layerSize];
		Arrays.fill(oob,0.0);

	}



	public Word2Vec(Collection<String> sentences) {
		createExpTable();
		this.buildVocab(sentences);
		this.sentences = new ArrayList<String>(sentences);
		oob = new double[layerSize];
		Arrays.fill(oob,0.0);

	}

	public Word2Vec(Collection<String> sentences,int minWordFrequency) {
		createExpTable();
		this.minWordFrequency = minWordFrequency;
		this.sentences = new ArrayList<String>(sentences);
		this.buildVocab(sentences);

		oob = new double[layerSize];
		Arrays.fill(oob,0.0);
	}


	public double[]	 getOob() {
		return oob;
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
			return null;
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

	public void train() {
		if(trainingSystem == null)
			trainingSystem = ActorSystem.create();
		log.info("Training word2vec multithreaded");
		MapFactory<String,Double> factory = new MapFactory<String,Double>() {

			private static final long serialVersionUID = 5447027920163740307L;

			@Override
			public Map<String, Double> buildMap() {
				return new java.util.concurrent.ConcurrentHashMap<String,Double>();
			}

		};
		final Counter<String> totalWords = new Counter<String>(factory);
		final CountDownLatch sentenceCounter = new CountDownLatch(sentences.size());
		if(syn0.rows != this.vocab.size())
			throw new IllegalStateException("We appear to be missing vectors here. Unable to train. Please ensure vectors were loaded properly.");

		for(final String sentence : sentences) {

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

		try {
			sentenceCounter.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		if(trainingSystem != null)
			trainingSystem.shutdown();

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


	public static class VocabWord implements Comparable<VocabWord>,Serializable {

		private static final long serialVersionUID = 2223750736522624256L;
		private double wordFrequency = 1;
		private int index = -1;
		private VocabWord left;
		private VocabWord right;
		private int code;
		private VocabWord parent;
		private int[] codes = null;
		private int[] points = null;

		//input layer to hidden layer, hidden layer to output layer
		private int layerSize = 200;
		/**
		 * 
		 * @param wordFrequency count of the word
		 * @param layerSize
		 */
		public VocabWord(double wordFrequency,int layerSize) {
			this.wordFrequency = wordFrequency;
			this.layerSize = layerSize;

		}


		private VocabWord() {}


		@Override
		public String toString() {
			return "VocabWord [wordFrequency=" + wordFrequency + ", index="
					+ index + ", left=" + left + ", right=" + right + ", code="
					+ code + ", codes=" + Arrays.toString(codes) + ", points=" + Arrays.toString(points)
					+ ", layerSize=" + layerSize + "]";
		}




		public void write(DataOutputStream dos) throws IOException {
			dos.writeDouble(wordFrequency);

		}

		public VocabWord read(DataInputStream dos,int layerSize) throws IOException {
			this.wordFrequency = dos.readDouble();
			this.layerSize = layerSize;
			return this;
		}




		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + code;
			result = prime * result + ((codes == null) ? 0 : codes.hashCode());
			result = prime * result + index;
			result = prime * result + layerSize;
			result = prime * result + ((left == null) ? 0 : left.hashCode());
			result = prime * result
					+ ((points == null) ? 0 : points.hashCode());
			result = prime * result + ((right == null) ? 0 : right.hashCode());
			long temp;
			temp = Double.doubleToLongBits(wordFrequency);
			result = prime * result + (int) (temp ^ (temp >>> 32));
			return result;
		}










		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			VocabWord other = (VocabWord) obj;
			if (code != other.code)
				return false;
			if (codes == null) {
				if (other.codes != null)
					return false;
			} else if (!codes.equals(other.codes))
				return false;
			if (index != other.index)
				return false;
			if (layerSize != other.layerSize)
				return false;
			if (left == null) {
				if (other.left != null)
					return false;
			} else if (!left.equals(other.left))
				return false;
			if (points == null) {
				if (other.points != null)
					return false;
			} else if (!points.equals(other.points))
				return false;
			if (right == null) {
				if (other.right != null)
					return false;
			} else if (!right.equals(other.right))
				return false;
			if (Double.doubleToLongBits(wordFrequency) != Double
					.doubleToLongBits(other.wordFrequency))
				return false;
			return true;
		}



		public int[] getCodes() {
			return codes;
		}

		public void setCodes(int[] codes) {
			this.codes = codes;
		}

		public int[] getPoints() {
			return points;
		}


		public void setPoints(int[] points) {
			this.points = points;
		}


		public void setWordFrequency(double wordFrequency) {
			this.wordFrequency = wordFrequency;
		}


		public int getLayerSize() {
			return layerSize;
		}


		public void setLayerSize(int layerSize) {
			this.layerSize = layerSize;
		}


		public VocabWord getParent() {
			return parent;
		}



		public void setParent(VocabWord parent) {
			this.parent = parent;
		}





		public int getCode() {
			return code;
		}





		public void setCode(int code) {
			this.code = code;
		}





		public VocabWord getLeft() {
			return left;
		}



		public void setLeft(VocabWord left) {
			this.left = left;
		}



		public VocabWord getRight() {
			return right;
		}



		public void setRight(VocabWord right) {
			this.right = right;
		}



		public void increment() {
			wordFrequency++;
		}


		public int getIndex() {
			return index;
		}

		public void setIndex(int index) {
			this.index = index;
		}

		public double getWordFrequency() {
			return wordFrequency;
		}


		@Override
		public int compareTo(VocabWord o) {
			return Double.compare(wordFrequency, o.wordFrequency);
		}

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


	public void loadModel(File file) throws Exception {
		log.info("Loading model from " + file.getAbsolutePath());
		try(DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
			int vocabSize = dis.readInt();
			int layerSize = dis.readInt();
			setLayerSize(layerSize);
			for(int i = 0; i < vocabSize; i++) {
				String word = dis.readUTF();
				wordIndex.add(word);
				vocab.put(word,new VocabWord().read(dis, layerSize));

			}
			syn0 = DoubleMatrix.zeros(vocabSize, layerSize);
			syn1 = DoubleMatrix.zeros(vocabSize, layerSize);

			syn0.in(dis);
			syn1.in(dis);

			dis.close();

		}
		catch(IOException e) {
			log.error("Unable to read file for loading model",e);
		}
	}

	public static double readFloat(InputStream is) throws IOException {
		byte[] bytes = new byte[4];
		is.read(bytes);
		return getFloat(bytes);
	}

	/**
	 *
	 * 
	 * @param b
	 * @return
	 */
	public static double getFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}




	public void loadTextModel(File file) throws IOException {
		List<String> list = FileUtils.readLines(file);
		this.buildVocab(list);
		this.sentences = new ArrayList<String>(list);
		this.train();

	}

	public void loadBinary(File file) throws IOException {
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(
				file)))) {
			words = dis.readInt();
			size = dis.readInt();
			this.layerSize = size;
			double len = 0;
			double vector = 0;
			syn0 = DoubleMatrix.zeros(words, size);
			syn1 = DoubleMatrix.zeros(words,size);

			String key = null;
			double[] value = null;
			for (int i = 0; i < words; i++) {
				key = dis.readUTF();
				value = new double[size];
				for (int j = 0; j < size; j++) {
					vector = dis.readDouble();
					len += vector * vector;
					value[j] = vector;
				}

				len = Math.sqrt(len);

				for (int j = 0; j < size; j++) 
					value[j] /= len;
				wordIndex.add(key);
				syn0.putRow(i,new DoubleMatrix(value));

			}

		}
	}

	private void loadGoogleVocab(String path) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		String temp = null;
		this.wordIndex = new Index();
		vocab.clear();
		while((temp = reader.readLine()) != null) {
			String[] split = temp.split(" ");
			if(split[0].equals("</s>"))
				continue;

			int freq = Integer.parseInt(split[1]);
			VocabWord realWord = new VocabWord(freq,layerSize);
			realWord.setIndex(vocab.size());
			this.vocab.put(split[0], realWord);
			wordIndex.add(split[0]);
		}
		reader.close();
	}


	public void loadGoogleText(String path,String vocabPath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		String temp = null;
		boolean first = true;
		Integer vectorSize = null;
		Integer rows = null;
		int currRow = 0;
		while((temp = reader.readLine()) != null) {
			if(first) {
				String[] split = temp.split(" ");
				rows = Integer.parseInt(split[0]);
				vectorSize = Integer.parseInt(split[1]);
				this.layerSize = vectorSize;
				syn0 = new DoubleMatrix(rows - 1,vectorSize);
				first = false;
			}

			else {
				StringTokenizer tokenizer = new StringTokenizer(temp);
				double[] vec = new double[layerSize];
				int count = 0;
				String word = tokenizer.nextToken();
				if(word.equals("</s>"))
					continue;

				while(tokenizer.hasMoreTokens()) {
					vec[count++] = Double.parseDouble(tokenizer.nextToken());
				}
				syn0.putRow(currRow, new DoubleMatrix(vec));
				currRow++;

			}
		}
		reader.close();

		loadGoogleVocab(vocabPath);

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



	public void buildVocab(Collection<String> sentences) {
		Map<String,VocabWord> vocab = new HashMap<String,VocabWord>();

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
				VocabWord word = vocab.get(token);

				//this will also increment the
				//vocab word at the final level
				//due to the reference being the same
				if(word != null)
					word.increment();
				else {
					word = new VocabWord(1.0,layerSize);
					vocab.put(token,word);
				}
				//note that for purposes of word frequency, the 
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(word.getWordFrequency() >= minWordFrequency) {
					if(!this.vocab.containsKey(token)) {
						word.setIndex(this.vocab.size());
						this.vocab.put(token, word);
						wordIndex.add(token);

					}


				}

			}

		}

		log.info("Building binary tree");
		buildBinaryTree();
		log.info("Resetting weights");
		resetWeights();
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

	public static double sigmoid(double x) {
		return 1f / (1f + Math.pow(Math.E, -x));
	}

	public static DoubleMatrix sigmoid(DoubleMatrix x) {
		DoubleMatrix matrix = new DoubleMatrix(x.rows,x.columns);
		for(int i = 0; i < matrix.length; i++)
			matrix.put(i, 1f / (1f + Math.pow(Math.E, -x.get(i))));

		return matrix;
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
		DoubleMatrix l1 = syn0.getRow(w2.index);
		DoubleMatrix l2a = syn1.getRows(w1.points);
		DoubleMatrix fa = sigmoid(MatrixUtil.dot(l1, l2a.transpose()));
		// ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
		DoubleMatrix ga = DoubleMatrix.ones(fa.length).sub(toMatrix(w1.codes)).sub(fa).mul(alpha);
		DoubleMatrix outer = ga.mmul(l1);
		for(int i = 0; i < w1.points.length; i++) {
			DoubleMatrix toAdd = l2a.getRow(i).add(outer.getRow(i));
			syn1.putRow(w1.points[i],toAdd);
		}

		DoubleMatrix updatedInput = l1.add(MatrixUtil.dot(ga, l2a));
		syn0.putRow(w2.index,updatedInput);
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


	private DoubleMatrix toMatrix(int [] codes) {
		double[] ret = new double[codes.length];
		for(int i = 0; i < codes.length; i++)
			ret[i] = codes[i];
		return new DoubleMatrix(ret);
	}


	private void buildBinaryTree() {
		PriorityQueue<VocabWord> heap = new PriorityQueue<VocabWord>(vocab.values());
		int i = 0;
		while(heap.size() > 1) {
			VocabWord min1 = heap.poll();
			VocabWord min2 = heap.poll();


			VocabWord add = new VocabWord(min1.wordFrequency + min2.wordFrequency,layerSize);
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
			if(node.index < vocab.size()) {
				node.setCodes(codes);
				node.setPoints(points);
			}
			else {
				int[] copy = plus(points,node.index - vocab.size());
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
		copy[addTo.length] =add;
		return copy;
	}


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


	public double similarity(String word,String word2) {
		if(word.equals(word2))
			return 1.0;

		double[] vector = getWordVector(word);
		double[] vector2 = getWordVector(word2);
		if(vector == null || vector2 == null)
			return -1;
		DoubleMatrix matrix = new DoubleMatrix(vector);
		DoubleMatrix matrix2 = new DoubleMatrix(vector2);
		double dot = matrix.dot(matrix2);
		double mag1 = magnitude(matrix);
		double mag2 = magnitude(matrix2);
		return dot / (mag1 * mag2);
	}

	private static double magnitude(DoubleMatrix vec) { 
		double sum_mag = 0; 
		for(int i = 0; i < vec.length;i++) 
			sum_mag = sum_mag + vec.get(i) * vec.get(i); 

		return Math.sqrt(sum_mag); 
	} 

	public static double cosine(DoubleMatrix matrix) {
		//1.0 * math.sqrt(sum(val * val for val in vec1.itervalues()))
		return 1 * Math.sqrt(MatrixFunctions.pow(matrix, 2).sum());
	}


	public static DoubleMatrix unitVec(DoubleMatrix matrix) {
		double norm2 = matrix.norm2();
		return SimpleBlas.scal(1 / norm2, matrix);
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


}
