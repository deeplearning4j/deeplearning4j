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
public class UIStatusReportingListener implements EarlyStoppingListener, IterationListener {

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

    private UICandidateStatusListener uiListener;

    private boolean invoked = false;
    private long lastReportTime = 0;
    private int recordEveryNthScore = 1;
    private long scoreCount = 0;
    private List<Double> scoreList = new ArrayList<>(MAX_SCORE_COMPONENTS);
    private List<Long> iterationList = new ArrayList<>(MAX_SCORE_COMPONENTS);
    private List<Pair<Integer,Double>> scoreVsEpochEarlyStopping = new ArrayList<>();

    private RenderableComponent config;


    public UIStatusReportingListener(UICandidateStatusListener listener){
        this.uiListener = listener;
    }


    @Override
    public void onStart(EarlyStoppingConfiguration esConfig, MultiLayerNetwork net) {
        if(config == null) createConfigComponent(net);
        postReport(Status.Running);
    }

    @Override
    public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, MultiLayerNetwork net) {
        if(config == null) createConfigComponent(net);
        scoreVsEpochEarlyStopping.add(new Pair<>(epochNum,score));

        postReport(Status.Running, createEarlyStoppingScoreVsEpochChart());
    }

    @Override
    public void onCompletion(EarlyStoppingResult esResult) {
        if(config == null) createConfigComponent(esResult.getBestModel());

        //Final status update: including early stopping results
        String[][] table = new String[][]{
                {"Termination reason:", esResult.getTerminationReason().toString()},
                {"Termination details:", esResult.getTerminationDetails()},
                {"Best model epoch:", String.valueOf(esResult.getBestModelEpoch())},
                {"Best model score:", String.valueOf(esResult.getBestModelScore())},
                {"Total epochs:", String.valueOf(esResult.getTotalEpochs())}
        };
        RenderableComponent rcTable = new RenderableComponentTable("Early Stopping",null,table);

        if(esResult.getTerminationReason() == EarlyStoppingResult.TerminationReason.Error){
            postReport(Status.Failed, rcTable);
        } else {
            postReport(Status.Complete, createEarlyStoppingScoreVsEpochChart(), rcTable);
        }
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

        RenderableComponent esScoreVsEpoch = new RenderableComponentLineChart.Builder()
                .addSeries("Score vs. Epoch",x,y)
                .title("Early Stopping: Score vs. Epoch")
                .build();
        return esScoreVsEpoch;
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
        if(config == null && model instanceof MultiLayerNetwork) createConfigComponent((MultiLayerNetwork)model);

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
            postReport(Status.Running);
        }
    }

    private void createConfigComponent(MultiLayerNetwork network){
        config = new RenderableComponentString(network.getLayerWiseConfigurations().toString());
    }

    public void postReport(Status status, RenderableComponent... additionalComponents){

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

        RenderableComponent scoreVsIterGraph = new RenderableComponentLineChart.Builder()
                .addSeries("Minibatch Score vs. Iteration",x,y)
                .title("Score vs. Iteration").build();

//        uiListener.reportStatus(status,config,scoreVsIterGraph);
        RenderableComponent[] rcs = new RenderableComponent[2 + (additionalComponents != null ? additionalComponents.length : 0)];
        rcs[0] = new RenderableComponentAccordionDecorator("Network Configuration",true,config);
        rcs[1] = scoreVsIterGraph;
        i = 2;
        for(RenderableComponent c : additionalComponents){
            rcs[i++] = c;
        }

        uiListener.reportStatus(status,rcs);

        lastReportTime = System.currentTimeMillis();
    }
}
