package org.deeplearning4j.scaleout.job;

import java.io.Serializable;

/**
 * Created by agibsonccc on 11/25/14.
 */
public class Job implements Serializable {
    private Serializable work;
    private Serializable result;
    private String workerId;




    public Job(Serializable work,String workerId) {
        this.work = work;
        this.workerId = workerId;
    }

    public void setWorkerId(String workerId) {
        this.workerId = workerId;
    }

    public String workerId() {
        return workerId;
    }

    public <E extends Serializable> E get(Class<E> clazz) {
        return clazz.cast(work);
    }

    public <E extends Serializable> E result(Class<E> clazz) {
        return clazz.cast(result);
    }

    public synchronized Serializable getWork() {
        return work;
    }

    public synchronized void setWork(Serializable work) {
        this.work = work;
    }

    public synchronized Serializable getResult() {
        return result;
    }

    public void setResult(Serializable result) {
        this.result = result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Job)) return false;

        Job job = (Job) o;

        if (result != null ? !result.equals(job.result) : job.result != null) return false;
        if (!work.equals(job.work)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result1 = work.hashCode();
        result1 = 31 * result1 + (result != null ? result.hashCode() : 0);
        return result1;
    }
}
