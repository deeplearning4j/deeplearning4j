/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.word2vec;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.util.concurrent.AtomicDouble;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


/**
 * Intermediate layers of the neural network
 *
 * @author Adam Gibson
 */
public  class VocabWord extends SequenceElement implements Serializable {

	private static final long serialVersionUID = 2223750736522624256L;
	//used in comparison when building the huffman tree
	private AtomicDouble wordFrequency = new AtomicDouble(0);
	private int index = -1;
	private List<Integer> codes = new ArrayList<>();
	//for my sanity
	private String word;
	@Getter @Setter private INDArray historicalGradient;
	private List<Integer> points = new ArrayList<>();
    private int codeLength = 0;

    /*
        Used for Joint/Distributed vocabs mechanics
     */
	@Getter @Setter protected Long vocabId;

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


    public String getLabel() {
        return this.word;
    }

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
		if(wordFrequency == null)
            return 0.0;

        return wordFrequency.get();
	}

    public List<Integer> getCodes() {
        return codes;
    }

    public void setCodes(List<Integer> codes) {
        this.codes = codes;
    }



	public double getGradient(int index, double g) {
		if(historicalGradient == null) {
			historicalGradient = Nd4j.zeros(getCodes().size());
		}

		double pow =  Math.pow(g,2);
		historicalGradient.putScalar(index, historicalGradient.getDouble(index) + pow);
		double sqrt =  FastMath.sqrt(historicalGradient.getDouble(index));
		double abs = FastMath.abs(g) / (sqrt + 1e-6f);
		double ret = abs * 1e-1f;
		return ret;

	}

    public List<Integer> getPoints() {
        return points;
    }

    public void setPoints(List<Integer> points) {
        this.points = points;
    }

    public int getCodeLength() {
        return codeLength;
    }

    public void setCodeLength(int codeLength) {
        this.codeLength = codeLength;
        if(codes.size() < codeLength) {
            for(int i = 0; i < codeLength; i++)
                codes.add(0);
        }

        if(points.size() < codeLength) {
            for(int i = 0; i < codeLength; i++)
                points.add(0);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        VocabWord vocabWord = (VocabWord) o;

        if (codeLength != vocabWord.codeLength) return false;
        if (index != vocabWord.index) return false;
        if (!codes.equals(vocabWord.codes)) return false;
        if (historicalGradient != null ? !historicalGradient.equals(vocabWord.historicalGradient) : vocabWord.historicalGradient != null)
            return false;
        if (!points.equals(vocabWord.points)) return false;
        if (!word.equals(vocabWord.word)) return false;
        return wordFrequency.get() == vocabWord.wordFrequency.get();

    }

    @Override
    public int hashCode() {
        int result = wordFrequency.hashCode();
        result = 31 * result + index;
        result = 31 * result + codes.hashCode();
        result = 31 * result + word.hashCode();
        result = 31 * result + (historicalGradient != null ? historicalGradient.hashCode() : 0);
        result = 31 * result + points.hashCode();
        result = 31 * result + codeLength;
        return result;
    }

    @Override
    public String toString() {
        return "VocabWord{" +
                "wordFrequency=" + wordFrequency +
                ", index=" + index +
                ", codes=" + codes +
                ", word='" + word + '\'' +
                ", historicalGradient=" + historicalGradient +
                ", points=" + points +
                ", codeLength=" + codeLength +
                '}';
    }


}

