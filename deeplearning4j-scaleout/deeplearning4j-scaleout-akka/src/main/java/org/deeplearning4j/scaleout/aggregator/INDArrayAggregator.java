package org.deeplearning4j.scaleout.aggregator;

import org.deeplearning4j.scaleout.job.Job;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * INDArray averager
 *
 * @author Adam Gibson
 */
public class INDArrayAggregator extends WorkAccumulator {
    private INDArray averaged;
    private static Logger log = LoggerFactory.getLogger(INDArrayAggregator.class);

    @Override
    public void accumulate(Job toAccumulate) {
        if(toAccumulate.getResult() == null || !(toAccumulate.getResult() instanceof INDArray)) {
            log.warn("Not accumulating result: must be of type INDArray and not null");
            return;
        }

        INDArray arr = (INDArray) toAccumulate.getResult();
        seenSoFar++;
        if(averaged == null) {
            this.averaged = arr;
        }

        else {
            averaged.addi(arr);
        }
    }

    @Override
    public Job aggregate() {
        if(averaged == null)
            return empty();
        Job ret =  new Job(averaged.div(seenSoFar),"");
        seenSoFar = 0.0;
        return ret;
    }
}
