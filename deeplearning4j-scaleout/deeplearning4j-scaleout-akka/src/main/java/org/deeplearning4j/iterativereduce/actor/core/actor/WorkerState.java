package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.io.Serializable;

import akka.actor.ActorRef;

public class WorkerState implements Serializable {

	private static final long serialVersionUID = 6984546372310389146L;
	private String workerId;
	private boolean isAvailable = true;
	private ActorRef ref;
	
	
	
	
	public WorkerState(String workerId, ActorRef ref) {
		super();
		this.workerId = workerId;
		this.ref = ref;
	}
	
	
	public  String getWorkerId() {
		return workerId;
	}
	public  void setWorkerId(String workerId) {
		this.workerId = workerId;
	}
	public  boolean isAvailable() {
		return isAvailable;
	}
	public  void setAvailable(boolean isAvailable) {
		this.isAvailable = isAvailable;
	}
	public  ActorRef getRef() {
		return ref;
	}
	public  void setRef(ActorRef ref) {
		this.ref = ref;
	}


	@Override
	public String toString() {
		return "WorkerState [workerId=" + workerId + ", isAvailable="
				+ isAvailable + "]";
	}
	
	

}
