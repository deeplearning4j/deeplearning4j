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

package org.deeplearning4j.scaleout.perform.models.glove;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.glove.CoOccurrences;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.scaleout.aggregator.JobAggregator;
import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.List;

/**
 * Base line word 2 vec performer
 *
 * @author Adam Gibson
 */
public class GlovePerformer implements WorkerPerformer {


    public final static String NAME_SPACE = "org.deeplearning4j.scaleout.perform.models.glove";
    public final static String VECTOR_LENGTH = NAME_SPACE + ".length";
    public final static String NUM_WORDS = NAME_SPACE + ".numwords";
    public final static String TABLE = NAME_SPACE + ".table";
    public final static String ALPHA = NAME_SPACE + ".alpha";
    public final static String ITERATIONS = NAME_SPACE + ".iterations";
    public final static String X_MAX = NAME_SPACE + ".xmax";
    public final static String MAX_COUNT = NAME_SPACE + ".maxcount";
    public final static String LOOKUPTABLE_SIZE = NAME_SPACE + ".lookuptablesize";
    private StateTracker stateTracker;
    private double xMax = 0.75;
    private static final Logger log = LoggerFactory.getLogger(GlovePerformer.class);
    private CoOccurrences coOccurrences;
    private double maxCount = 100;
    private int[] lookupTableSize;
    private int[] biasShape;


    public GlovePerformer(StateTracker stateTracker) {
        this.stateTracker = stateTracker;
    }


    public GlovePerformer() {}

    @Override
    public void perform(Job job) {

        if(job.getWork() instanceof GloveWork) {
            GloveWork work = (GloveWork) job.getWork();

            if(work == null)
                return;

            List<Pair<VocabWord,VocabWord>> sentences = work.getCoOccurrences();
            for(Pair<VocabWord,VocabWord> coc : sentences)
                iterateSample(work,coc.getFirst(),coc.getSecond(),coOccurrences.count(coc.getFirst().getWord(),coc.getSecond().getWord()));
            job.setResult((Serializable) Arrays.asList(work.addDeltas()));

        }





    }

    @Override
    public void update(Object... o) {

    }

    @Override
    public void setup(Configuration conf) {
        xMax = conf.getFloat(X_MAX,0.75f);
        maxCount = conf.getFloat(MAX_COUNT,100);
        lookupTableSize = getInts(conf,LOOKUPTABLE_SIZE);
        biasShape = new int[]{lookupTableSize[1]};

        String connectionString = conf.get(STATE_TRACKER_CONNECTION_STRING);


        log.info("Creating state tracker with connection string "+  connectionString);
        if(stateTracker == null)
            try {
                stateTracker = new HazelCastStateTracker(connectionString);
            } catch (Exception e) {
                e.printStackTrace();
            }



        coOccurrences = stateTracker.get(GloveJobIterator.CO_OCCURRENCES);
        if(coOccurrences == null)
            throw new IllegalStateException("Please specify co occurrences");
    }


    private int[] getInts(Configuration conf,String key) {
        String[] strs = conf.getStrings(key);
        int[] ret = new int[strs.length];
        for(int i = 0; i < ret.length; i++)
            ret[i] = Integer.parseInt(strs[i]);
        return ret;
    }

    /**
     * Configure the configuration based on the table and index
     * @param table the table
     * @param index the index
     * @param conf the configuration
     */
    public static void configure(GloveWeightLookupTable table,InvertedIndex index,Configuration conf) {
        if(table.getSyn0() == null)
            throw new IllegalStateException("Unable to configure glove: missing look up table size. Please call table.resetWeights() first");
        conf.setInt(VECTOR_LENGTH, table.layerSize());
        conf.setFloat(ALPHA,(float) table.getLr().get());
        conf.setStrings(LOOKUPTABLE_SIZE,String.valueOf(table.getSyn0().rows()),String.valueOf(table.getSyn0().columns()));
        conf.setLong(NUM_WORDS, index.totalWords());
        conf.set(JobAggregator.AGGREGATOR, GloveJobAggregator.class.getName());
        conf.set(WorkerPerformerFactory.WORKER_PERFORMER,GlovePerformerFactory.class.getName());
        table.resetWeights();
        if(table.getNegative() > 0) {
            ByteArrayOutputStream bis = new ByteArrayOutputStream();
            try {
                DataOutputStream ois = new DataOutputStream(bis);
                Nd4j.write(table.getTable(),ois);
            } catch (IOException e) {
                e.printStackTrace();
            }
            conf.set(GlovePerformer.TABLE,new String(bis.toByteArray()));

        }
    }




    /**
     * glove iteration
     * @param w1 the first word
     * @param w2 the second word
     * @param score the weight learned for the particular co occurrences
     */
    public   double iterateSample(GloveWork work,VocabWord w1, VocabWord w2,double score) {
        INDArray w1Vector = work.getOriginalVectors().get(w1.getWord());
        INDArray w2Vector = work.getOriginalVectors().get(w2.getWord());
        //prediction: input + bias

        //w1 * w2 + bias
        double prediction = Nd4j.getBlasWrapper().dot(w1Vector,w2Vector);
        prediction +=  work.getBiases().get(w1.getWord()) + work.getBiases().get(w2.getWord());

        double weight = Math.pow(Math.min(1.0,(score / maxCount)),xMax);

        double fDiff = score > xMax ? prediction :  weight * (prediction - Math.log(score));


        //amount of change
        double gradient =  fDiff;

        //note the update step here: the gradient is
        //the gradient of the OPPOSITE word
        //for adagrad we will use the index of the word passed in
        //for the gradient calculation we will use the context vector


        update(work,w1,w1Vector,w2Vector,gradient);
        update(work,w2,w2Vector,w1Vector,gradient);
        return fDiff;



    }


    private void update(GloveWork gloveWork,VocabWord w1,INDArray wordVector,INDArray contextVector,double gradient) {
        //gradient for word vectors
        INDArray grad1 = contextVector.mul(gradient);
        //adagrad will be one row
        INDArray update = gloveWork.getAdaGrad(w1.getWord()).getGradient(grad1, 0, lookupTableSize);

        //update vector
        wordVector.subi(update);

        double w1Bias = gloveWork.getBias(w1.getWord());
        //adagrad will be one number
        double biasGradient = gloveWork.getBiasAdaGrad(w1.getWord()).getGradient(gradient,0, biasShape);
        double update2 = w1Bias - biasGradient;
        gloveWork.updateBias(w1.getWord(),update2);


    }






}
