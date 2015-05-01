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

package org.deeplearning4j.scaleout.actor;

import org.canova.api.conf.Configuration;
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
