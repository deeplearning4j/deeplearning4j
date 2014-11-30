package org.deeplearning4j.scaleout.perform.models.word2vec;

import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Handles creating a total word2vec model
 * @author Adam Gibson
 */
public class Word2VecJobAggregator implements JobAggregator {
    private List<Word2VecResult> work = new ArrayList<>();


    @Override
    public void accumulate(Job job) {
        if(job.getResult() instanceof Word2VecResult) {
            Word2VecResult work = (Word2VecResult) job.getResult();
            this.work.add(work);
        }
        else if(job.getResult() instanceof Collection) {
            Collection<Word2VecResult> coll = (Collection<Word2VecResult>) job.getResult();
            work.addAll(coll);
        }

    }

    @Override
    public Job aggregate() {
        Job ret =  new Job("","");
        ret.setResult((Serializable) work);
        return ret;
    }

    @Override
    public void init(Configuration conf) {

    }
}
