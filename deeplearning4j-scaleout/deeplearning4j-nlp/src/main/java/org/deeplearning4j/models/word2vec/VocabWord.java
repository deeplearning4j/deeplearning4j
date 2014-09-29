package org.deeplearning4j.models.word2vec;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.util.concurrent.AtomicDouble;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;





/**
 * Intermediate layers of the neural network
 *
 * @author Adam Gibson
 */
public  class VocabWord implements Comparable<VocabWord>,Serializable {

	private static final long serialVersionUID = 2223750736522624256L;
	//used in comparison when building the huffman tree
	private AtomicDouble wordFrequency = new AtomicDouble(0);
	private int index = -1;
	//children of the binary tree
	private VocabWord left;
	private VocabWord right;
	private VocabWord parent;
	private int[] codes = new int[40];
	//for my sanity
	private String word;
	public final static String PARENT_NODE = "parent";
	private INDArray historicalGradient;
	private int[] points = new int[40];
    private int codeLength = 0;
	

	public static VocabWord none() {
		return new VocabWord(0,"none");
	}

	/**
	 *
	 * @param wordFrequency count of the word

	 */
	public VocabWord(double wordFrequency,String word) {
		this.wordFrequency.set(wordFrequency);
		if(word == null || word.isEmpty())
			throw new IllegalArgumentException("Word must not be null or empty");
		this.word = word;

	}


	public VocabWord() {}


	public void write(DataOutputStream dos) throws IOException {
		dos.writeDouble(wordFrequency.get());

	}

	public VocabWord read(DataInputStream dos) throws IOException {
		this.wordFrequency.set(dos.readDouble());
		return this;
	}



	public String getWord() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}





	public int[] getCodes() {
		return codes;
	}

	public void setCodes(int[] codes) {
		this.codes = codes;
	}



	public void setParent(VocabWord parent) {
		this.parent = parent;
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
		increment(1);
	}

	public void increment(int by) {
		wordFrequency.getAndAdd(by);
	}


	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}

	public double getWordFrequency() {
		return wordFrequency.get();
	}

	
	
	@Override
	public int compareTo(VocabWord o) {
		return Double.compare(wordFrequency.get(), o.wordFrequency.get());
	}

	public float getLearningRate(int index,float g) {
		if(historicalGradient == null) {
			historicalGradient = Nd4j.zeros(getCodes().length);
		}

		float pow = (float) Math.pow(g,2);
		historicalGradient.putScalar(index, historicalGradient.get(index) + pow);
		float sqrt = (float) FastMath.sqrt(historicalGradient.get(index));
		float abs = FastMath.abs(g) / (sqrt + 1e-6f);
		float ret = abs * 1e-1f;
		return ret;

	}

	public int[] getPoints() {
		return points;
	}

	public void setPoints(int[] points) {
		this.points = points;
	}

    public int getCodeLength() {
        return codeLength;
    }

    public void setCodeLength(int codeLength) {
        this.codeLength = codeLength;
    }

    @Override
    public String toString() {
        return "VocabWord{" +
                "wordFrequency=" + wordFrequency +
                ", index=" + index +
                ", codes=" + Arrays.toString(codes) +
                ", word='" + word + '\'' +
                ", historicalGradient=" + historicalGradient +
                ", points=" + Arrays.toString(points) +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof VocabWord)) return false;

        VocabWord vocabWord = (VocabWord) o;

        if (index != vocabWord.index) return false;
        if (!Arrays.equals(codes, vocabWord.codes)) return false;
        if (historicalGradient != null ? !historicalGradient.equals(vocabWord.historicalGradient) : vocabWord.historicalGradient != null)
            return false;
        if (left != null ? !left.equals(vocabWord.left) : vocabWord.left != null) return false;
        if (parent != null ? !parent.equals(vocabWord.parent) : vocabWord.parent != null) return false;
        if (!Arrays.equals(points, vocabWord.points)) return false;
        if (right != null ? !right.equals(vocabWord.right) : vocabWord.right != null) return false;
        if (word != null ? !word.equals(vocabWord.word) : vocabWord.word != null) return false;
        if (wordFrequency != null ? !wordFrequency.equals(vocabWord.wordFrequency) : vocabWord.wordFrequency != null)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = wordFrequency != null ? wordFrequency.hashCode() : 0;
        result = 31 * result + index;
        result = 31 * result + (left != null ? left.hashCode() : 0);
        result = 31 * result + (right != null ? right.hashCode() : 0);
        result = 31 * result + (parent != null ? parent.hashCode() : 0);
        result = 31 * result + (codes != null ? Arrays.hashCode(codes) : 0);
        result = 31 * result + (word != null ? word.hashCode() : 0);
        result = 31 * result + (historicalGradient != null ? historicalGradient.hashCode() : 0);
        result = 31 * result + (points != null ? Arrays.hashCode(points) : 0);
        return result;
    }
}

