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
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.io.Serializable;


/**
 * Intermediate layers of the neural network
 *
 * @author Adam Gibson
 */
public  class VocabWord extends SequenceElement implements Serializable {

	private static final long serialVersionUID = 2223750736522624256L;

	//for my sanity
	private String word;

    /*
        Used for Joint/Distributed vocabs mechanics
     */
	@Getter @Setter protected Long vocabId;
	@Getter @Setter protected Long affinityId;

	public static VocabWord none() {
		return new VocabWord(0,"none");
	}

	/**
	 *
	 * @param wordFrequency count of the word

	 */
	public VocabWord(double wordFrequency,String word) {
		if(word == null || word.isEmpty())
			throw new IllegalArgumentException("Word must not be null or empty");
		this.word = word;
        this.elementFrequency.set(wordFrequency);
	}


	public VocabWord() {}


    public String getLabel() {
        return this.word;
    }
/*
	public void write(DataOutputStream dos) throws IOException {
		dos.writeDouble(this.elementFrequency.get());
	}

	public VocabWord read(DataInputStream dos) throws IOException {
		this.elementFrequency.set(dos.readDouble());
		return this;
	}

*/

	public String getWord() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}

    /*
	public void increment() {
		increment(1);
	}

	public void increment(int by) {
		this.elementFrequency.getAndAdd(by);
	}


	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}


	public double getWordFrequency() {
		if(this.elementFrequency == null)
            return 0.0;

        return elementFrequency.get();
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
    */

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        VocabWord vocabWord = (VocabWord) o;

        return this.word.equals(vocabWord.getWord());
        /*
        if (codeLength != vocabWord.codeLength) return false;
        if (index != vocabWord.index) return false;
        if (!codes.equals(vocabWord.codes)) return false;
        if (historicalGradient != null ? !historicalGradient.equals(vocabWord.historicalGradient) : vocabWord.historicalGradient != null)
            return false;
        if (!points.equals(vocabWord.points)) return false;
        if (!word.equals(vocabWord.word)) return false;
        return this.elementFrequency.get() == vocabWord.elementFrequency.get();
        */
    }

    @Override
    public int hashCode() {
        int result = this.word == null ? 0 : this.word.hashCode(); //this.elementFrequency.hashCode();
        /*result = 31 * result + index;
        result = 31 * result + codes.hashCode();
        result = 31 * result + word.hashCode();
        result = 31 * result + (historicalGradient != null ? historicalGradient.hashCode() : 0);
        result = 31 * result + points.hashCode();
        result = 31 * result + codeLength;*/
        return result;
    }

    @Override
    public String toString() {
        return "VocabWord{" +
                "wordFrequency=" + this.elementFrequency +
                ", index=" + index +
                ", codes=" + codes +
                ", word='" + word + '\'' +
                ", historicalGradient=" + historicalGradient +
                ", points=" + points +
                ", codeLength=" + codeLength +
                '}';
    }

	@Override
	public String toJSON() {
		return null;
	}


}

