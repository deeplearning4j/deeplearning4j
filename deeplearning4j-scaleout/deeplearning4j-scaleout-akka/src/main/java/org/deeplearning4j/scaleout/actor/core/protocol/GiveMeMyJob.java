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

package org.deeplearning4j.scaleout.actor.core.protocol;

import org.deeplearning4j.scaleout.job.Job;

import java.io.Serializable;

public class GiveMeMyJob implements Serializable {

	private String id;
	private Job job;
	public GiveMeMyJob(String id,Job job) {
		super();
		this.id = id;
		this.job = job;
	}

	public  String getId() {
		return id;
	}

	public  void setId(String id) {
		this.id = id;
	}

	public synchronized Job getJob() {
		return job;
	}

	public synchronized void setJob(Job job) {
		this.job = job;
	}
	
	
	
}
