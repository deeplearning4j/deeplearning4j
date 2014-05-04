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


	public Job(String workerId, Serializable work) {
		super();
		this.workerId = workerId;
		this.work = work;
	}



    public Job(Job job) {
        this.done = job.done;
        this.workerId = job.workerId;
        this.work = job.work;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Job)) return false;

        Job job = (Job) o;

        if (done != job.done) return false;
        if (work != null ? !work.equals(job.work) : job.work != null) return false;
        if (workerId != null ? !workerId.equals(job.workerId) : job.workerId != null) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (done ? 1 : 0);
        result = 31 * result + (workerId != null ? workerId.hashCode() : 0);
        result = 31 * result + (work != null ? work.hashCode() : 0);
        return result;
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

    /**
     * Returns a string representation of the object. In general, the
     * {@code toString} method returns a string that
     * "textually represents" this object. The result should
     * be a concise but informative representation that is easy for a
     * person to read.
     * It is recommended that all subclasses override this method.
     * <p/>
     * The {@code toString} method for class {@code Object}
     * returns a string consisting of the name of the class of which the
     * object is an instance, the at-sign character `{@code @}', and
     * the unsigned hexadecimal representation of the hash code of the
     * object. In other words, this method returns a string equal to the
     * value of:
     * <blockquote>
     * <pre>
     * getClass().getName() + '@' + Integer.toHexString(hashCode())
     * </pre></blockquote>
     *
     * @return a string representation of the object.
     */
    @Override
    public String toString() {
        return super.toString();
    }
}
