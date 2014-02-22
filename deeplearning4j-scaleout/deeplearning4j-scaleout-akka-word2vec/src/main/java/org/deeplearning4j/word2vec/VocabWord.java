package org.deeplearning4j.word2vec;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Arrays;


public  class VocabWord implements Comparable<VocabWord>,Serializable {

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


	public VocabWord() {}


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

