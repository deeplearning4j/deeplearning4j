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
    //for my sanity
    private String word;

	public static VocabWord none() {
		return new VocabWord(0,"none");
	}

	/**
	 *
	 * @param wordFrequency count of the word

	 */
	public VocabWord(double wordFrequency,String word) {
		this.elementFrequency.set(wordFrequency);
		if(word == null || word.isEmpty())
			throw new IllegalArgumentException("Word must not be null or empty");
		this.word = word;

	}


	public VocabWord() {}

	@Override
    public String getLabel() {
        return this.word;
    }

	public void write(DataOutputStream dos) throws IOException {
		dos.writeDouble(elementFrequency.get());

	}

	public VocabWord read(DataInputStream dos) throws IOException {
		this.elementFrequency.set(dos.readDouble());
		return this;
	}



	public String getWord() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
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
        return elementFrequency.get() == vocabWord.elementFrequency.get();

    }

    @Override
    public int hashCode() {
        int result = elementFrequency.hashCode();
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
                "wordFrequency=" + elementFrequency +
                ", index=" + index +
                ", codes=" + codes +
                ", word='" + word + '\'' +
                ", historicalGradient=" + historicalGradient +
                ", points=" + points +
                ", codeLength=" + codeLength +
                '}';
    }


}

