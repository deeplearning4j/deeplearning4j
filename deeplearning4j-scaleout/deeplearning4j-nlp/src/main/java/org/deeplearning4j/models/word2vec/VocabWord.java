package org.deeplearning4j.models.word2vec;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.util.concurrent.AtomicDouble;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;




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
	private int code;
	private VocabWord parent;
	private int[] codes = null;
	//for my sanity
	private String word;
	public final static String PARENT_NODE = "parent";
	private INDArray historicalGradient;
	private List<VocabWord> connections;

	public static VocabWord none() {
		return new VocabWord(0,"none");
	}

	/**
	 *
	 * @param wordFrequency count of the word

	 */
	public VocabWord(double wordFrequency,String word) {
		this.wordFrequency.set(wordFrequency);
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




	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + code;
		result = prime * result + Arrays.hashCode(codes);
		result = prime * result
				+ ((connections == null) ? 0 : connections.hashCode());
		result = prime
				* result
				+ ((historicalGradient == null) ? 0 : historicalGradient
						.hashCode());
		result = prime * result + index;
		result = prime * result + ((left == null) ? 0 : left.hashCode());
		result = prime * result + ((parent == null) ? 0 : parent.hashCode());
		result = prime * result + ((right == null) ? 0 : right.hashCode());
		result = prime * result + ((word == null) ? 0 : word.hashCode());
		result = prime * result
				+ ((wordFrequency == null) ? 0 : wordFrequency.hashCode());
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
		if (!Arrays.equals(codes, other.codes))
			return false;
		if (connections == null) {
			if (other.connections != null)
				return false;
		} else if (!connections.equals(other.connections))
			return false;
		if (historicalGradient == null) {
			if (other.historicalGradient != null)
				return false;
		} else if (!historicalGradient.equals(other.historicalGradient))
			return false;
		if (index != other.index)
			return false;
		if (left == null) {
			if (other.left != null)
				return false;
		} else if (!left.equals(other.left))
			return false;
		if (parent == null) {
			if (other.parent != null)
				return false;
		} else if (!parent.equals(other.parent))
			return false;
		if (right == null) {
			if (other.right != null)
				return false;
		} else if (!right.equals(other.right))
			return false;
		if (word == null) {
			if (other.word != null)
				return false;
		} else if (!word.equals(other.word))
			return false;
		if (wordFrequency == null) {
			if (other.wordFrequency != null)
				return false;
		} else if (!wordFrequency.equals(other.wordFrequency))
			return false;
		return true;
	}

	public INDArray getHistoricalGradient() {
		return historicalGradient;
	}

	public void setHistoricalGradient(INDArray historicalGradient) {
		this.historicalGradient = historicalGradient;
	}

	public List<VocabWord> getConnections() {
		return connections;
	}

	public void setConnections(List<VocabWord> connections) {
		this.connections = connections;
	}

	public void setWordFrequency(AtomicDouble wordFrequency) {
		this.wordFrequency = wordFrequency;
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




	public void setWordFrequency(double wordFrequency) {
		this.wordFrequency.set(wordFrequency);
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


	public void createLinks() {
		VocabWord curr = this;
		if(connections != null)
			return;
		connections = new ArrayList<>();
		while((curr = curr.parent) != null) {
			connections.add(curr);
		}
		
		Collections.reverse(connections);
		
		codes = new int[connections.size()];
		
		for(int i = 1; i < codes.length; i++) {
			codes[i - 1] = connections.get(i).code;
		}
		
		codes[codes.length - 1] = code;
		
		
		
		

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


}

