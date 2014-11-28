package org.deeplearning4j.scaleout.actor;

import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class TestPerformer implements WorkerPerformer {
    private List<Job> jobs = new ArrayList<>();
    private Configuration conf;
    private boolean updateCalled = false;
    private boolean performCalled = false;

    @Override
    public void perform(Job job) {
        jobs.add(job);
        job.setResult("done");
        performCalled = true;
    }

    @Override
    public void update(Object... o) {
        updateCalled = true;


    }

    @Override
    public void setup(Configuration conf) {
        this.conf = conf;
    }

    public List<Job> getJobs() {
        return jobs;
    }

    public void setJobs(List<Job> jobs) {
        this.jobs = jobs;
    }

    public Configuration getConf() {
        return conf;
    }

    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    public boolean isUpdateCalled() {
        return updateCalled;
    }

    public void setUpdateCalled(boolean updateCalled) {
        this.updateCalled = updateCalled;
    }

    public boolean isPerformCalled() {
        return performCalled;
    }

    public void setPerformCalled(boolean performCalled) {
        this.performCalled = performCalled;
    }
}
