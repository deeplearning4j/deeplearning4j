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
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;


public class VocabMessage implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6474030463892664765L;
	private List<String> tokens;
	private AtomicLong changeTracker;
	
	public VocabMessage(List<String> tokens,AtomicLong changeTracker) {
		super();
		this.tokens = tokens;
		this.changeTracker = changeTracker;
	}
	
	



	public AtomicLong getChangeTracker() {
		return changeTracker;
	}





	public void setChangeTracker(AtomicLong changeTracker) {
		this.changeTracker = changeTracker;
	}


	
	public List<String> getTokens() {
		return tokens;
	}
	public void setTokens(List<String> tokens) {
		this.tokens = tokens;
	}
	
	
	
}
