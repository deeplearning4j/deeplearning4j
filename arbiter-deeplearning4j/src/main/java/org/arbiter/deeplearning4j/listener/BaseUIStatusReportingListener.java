/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.arbiter.deeplearning4j.listener;

import org.arbiter.optimize.runner.Status;
import org.arbiter.optimize.runner.listener.candidate.UICandidateStatusListener;
import org.arbiter.optimize.ui.components.*;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.*;

/**Listener designed to report status to Arbiter UI
 * Combines listener functionality for both early stopping AND iteration listeners
 */
public abstract class BaseUIStatusReportingListener<T extends Model> implements EarlyStoppingListener<T>, IterationListener {

    /** How frequently (maximum delay between reporting, in MS) should results be reported? This is necessary to keep
     * network traffic to a reasonable level.
     * onStart, onEpoch and onCompletion calls are exempt from this
     */
    public static final int MAX_REPORTING_FREQUENCY_MS = 5000; //Report at most every 5 seconds

    /** Score vs. iteration reporting: how many scores (maximum) should we report? This is necessary to keep
     * network traffic to a reasonable level.
     * When the number of reported scores exceeds this, the score history will be subsampled: i.e., report only
     * every 2nd score, then every 4th, then every 8th etc as required to keep total number of reported scores
     */
    public static final int MAX_SCORE_COMPONENTS = 4000;

    protected UICandidateStatusListener uiListener;

    protected boolean invoked = false;
    protected long lastReportTime = 0;
    protected int recordEveryNthScore = 1;
    protected long scoreCount = 0;
    protected List<Double> scoreList = new ArrayList<>(MAX_SCORE_COMPONENTS);
    protected List<Long> iterationList = new ArrayList<>(MAX_SCORE_COMPONENTS);
    protected List<Pair<Integer,Double>> scoreVsEpochEarlyStopping = new ArrayList<>();

    protected RenderableComponent config;


    public BaseUIStatusReportingListener(UICandidateStatusListener listener){
        this.uiListener = listener;
    }


    @Override
    public void onStart(EarlyStoppingConfiguration<T> esConfig, T net) {
        if(config == null) createConfigComponent(net);
        postReport(Status.Running,null);
    }

    @Override
    public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<T> esConfig, T net) {
        if(config == null) createConfigComponent(net);
        scoreVsEpochEarlyStopping.add(new Pair<>(epochNum,score));

        postReport(Status.Running, null, createEarlyStoppingScoreVsEpochChart());
    }

    @Override
    public void onCompletion(EarlyStoppingResult<T> esResult) {
        if(config == null) createConfigComponent(esResult.getBestModel());
    }

    private RenderableComponent createEarlyStoppingScoreVsEpochChart(){
        double[] x = new double[scoreVsEpochEarlyStopping.size()];
        double[] y = new double[scoreVsEpochEarlyStopping.size()];
        int i=0;
        for(Pair<Integer,Double> p : scoreVsEpochEarlyStopping){
            x[i] = p.getFirst();
            y[i] = p.getSecond();
            i++;
        }

        return new RenderableComponentLineChart.Builder()
                .addSeries("Score vs. Epoch",x,y)
                .title("Early Stopping: Score vs. Epoch")
                .build();
    }

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(config == null) createConfigComponent((T)model);

        double score = model.score();

        if(scoreList.size() <= MAX_SCORE_COMPONENTS){
            if(scoreCount % recordEveryNthScore == 0){
                //Record this score
                scoreList.add(score);
                iterationList.add(scoreCount);
            }
            scoreCount++;
        } else {
            //Subsample existing scores
            recordEveryNthScore *= 2;
            List<Double> newScoreList = new ArrayList<>(MAX_SCORE_COMPONENTS);
            List<Long> newIterationList = new ArrayList<>(MAX_SCORE_COMPONENTS);
            Iterator<Double> sIter = scoreList.iterator();
            Iterator<Long> iIter = iterationList.iterator();
            int i=0;
            while(sIter.hasNext()){
                //Keep every 2nd score/time pair
                if(i++ % 2 == 0){
                    newScoreList.add(sIter.next());
                    newIterationList.add(iIter.next());
                } else {
                    sIter.next();
                    iIter.next();
                }
            }

            scoreList = newScoreList;
            iterationList = newIterationList;
        }

        long currTime = System.currentTimeMillis();
        if(currTime - lastReportTime > MAX_REPORTING_FREQUENCY_MS ){
            //Post report
            postReport(Status.Running,null);
        }
    }

    protected abstract void createConfigComponent(T network);
//    {
//        config = new RenderableComponentString(network.getLayerWiseConfigurations().toString());
//    }

    public void postReport(Status status, EarlyStoppingResult<T> esResult, RenderableComponent... additionalComponents){

        //Create score vs. iteration graph:
        double[] x = new double[scoreList.size()];
        double[] y = new double[scoreList.size()];
        Iterator<Double> sIter = scoreList.iterator();
        Iterator<Long> iIter = iterationList.iterator();
        int i=0;
        while(sIter.hasNext() && i < x.length){
            y[i] = sIter.next();
            x[i] = iIter.next();
            i++;
        }

        List<RenderableComponent> components = new ArrayList<>();
        components.add(new RenderableComponentAccordionDecorator("Network Configuration",true,config));

        RenderableComponent scoreVsIterGraph = new RenderableComponentLineChart.Builder()
                .addSeries("Minibatch Score vs. Iteration",x,y)
                .title("Score vs. Iteration").build();
        components.add(scoreVsIterGraph);

        if(esResult != null){
            //Final status update: including early stopping results
            int bestEpoch = esResult.getBestModelEpoch();


            String[][] table = new String[][]{
                    {"Termination reason:", esResult.getTerminationReason().toString()},
                    {"Termination details:", esResult.getTerminationDetails()},
                    {"Best model epoch:", (bestEpoch < 0 ? "n/a" : String.valueOf(bestEpoch)) },
                    {"Best model score:", (bestEpoch < 0 ? "n/a" : String.valueOf(esResult.getBestModelScore())) },
                    {"Total epochs:", String.valueOf(esResult.getTotalEpochs())}
            };
            RenderableComponent rcTable = new RenderableComponentTable("Early Stopping",null,table);
            components.add(rcTable);
        }

        if(additionalComponents != null){
            Collections.addAll(components, additionalComponents);
        }

        uiListener.reportStatus(status,components.toArray(new RenderableComponent[components.size()]));

        lastReportTime = System.currentTimeMillis();
    }
}
