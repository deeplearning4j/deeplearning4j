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

package org.deeplearning4j.models.word2vec.actor;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicLong;


public class SentenceMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8132989189483837222L;
	private String sentence;
	private  AtomicLong changed;

    public SentenceMessage(String sentence,AtomicLong changed) {
		super();
		this.sentence = sentence;
		this.changed = changed;
	}

    public String getSentence() {
		return sentence;
	}
	public void setSentence(String sentence) {
		this.sentence = sentence;
	}
	public AtomicLong getChanged() {
		return changed;
	}
	public void setChanged(AtomicLong changed) {
		this.changed = changed;
	}
	
	

}
