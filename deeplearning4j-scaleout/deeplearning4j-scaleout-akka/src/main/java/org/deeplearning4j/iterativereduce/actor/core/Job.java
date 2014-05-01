package org.deeplearning4j.iterativereduce.actor.core;

import java.io.Serializable;
/**
 * A job represents a unit of work.
 * This is for communication between the master and workers
 * to track statuses
 * @author Adam Gibson
 *
 */
public class Job implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6762101539446662016L;
	private boolean done;
	private String workerId;
	private Serializable work;
	private boolean pretrain;
	
	public Job(String workerId, Serializable work,boolean pretrain) {
		this(false,workerId,work,pretrain);
	}
	
	public Job(boolean done, String workerId, Serializable work,boolean pretrain) {
		super();
		this.done = done;
		this.workerId = workerId;
		this.work = work;
		this.pretrain = pretrain;
	}



    public Job(Job job) {
        this.done = job.done;
        this.workerId = job.workerId;
        this.work = job.work;
        this.pretrain = job.pretrain;
    }


	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + (done ? 1231 : 1237);
		result = prime * result + (pretrain ? 1231 : 1237);
		result = prime * result + ((work == null) ? 0 : work.hashCode());
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
		Job other = (Job) obj;
		if (done != other.done)
			return false;
		if (pretrain != other.pretrain)
			return false;
		if (work == null) {
			if (other.work != null)
				return false;
		} else if (!work.equals(other.work))
			return false;
		if (workerId == null) {
			if (other.workerId != null)
				return false;
		} else if (!workerId.equals(other.workerId))
			return false;
		return true;
	}


    @Override
    public Job clone() {
        return new Job(this);
    }
	
	
	public boolean isDone() {
		return done;
	}
	public void setDone(boolean done) {
		this.done = done;
	}
	public String getWorkerId() {
		return workerId;
	}
	public void setWorkerId(String workerId) {
		this.workerId = workerId;
	}
	public Serializable getWork() {
		return work;
	}
	public void setWork(Serializable work) {
		this.work = work;
	}

	public  boolean isPretrain() {
		return pretrain;
	}

	public void setPretrain(boolean pretrain) {
		this.pretrain = pretrain;
	}

	@Override
	public String toString() {
		return "Job [done=" + done + ", workerId=" + workerId + ", pretrain="
				+ pretrain + "]";
	}
	
	
	
	

}
