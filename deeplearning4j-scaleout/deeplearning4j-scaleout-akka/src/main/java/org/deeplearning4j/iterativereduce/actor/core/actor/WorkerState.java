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


	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + (isAvailable ? 1231 : 1237);
		result = prime * result + ((ref == null) ? 0 : ref.hashCode());
		result = prime * result
				+ ((workerId == null) ? 0 : workerId.hashCode());
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
		WorkerState other = (WorkerState) obj;
		if (isAvailable != other.isAvailable)
			return false;
		if (ref == null) {
			if (other.ref != null)
				return false;
		} else if (!ref.equals(other.ref))
			return false;
		if (workerId == null) {
			if (other.workerId != null)
				return false;
		} else if (!workerId.equals(other.workerId))
			return false;
		return true;
	}
	
	

}
