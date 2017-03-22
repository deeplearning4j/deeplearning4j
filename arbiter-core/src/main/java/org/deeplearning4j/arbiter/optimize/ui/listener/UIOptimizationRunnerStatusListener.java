/*-
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
package org.deeplearning4j.arbiter.optimize.ui.listener;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.Status;
import org.deeplearning4j.arbiter.optimize.runner.listener.runner.OptimizationRunnerStatusListener;
import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.LengthUnit;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.ChartScatter;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;

import java.awt.*;
import java.util.*;
import java.util.List;

/**
 * A listener for the optimization runner that reports results to the UI
 *
 * @author Alex Black
 */
public class UIOptimizationRunnerStatusListener implements OptimizationRunnerStatusListener {

    private ArbiterUIServer server;
    private long startTime;
    private List<Double> bestScores = new ArrayList<>();
    private List<Long> bestScoreStartTimes = new ArrayList<>();
    private double lastBestScore;
    private long lastBestScoreTime = 0;

    private StyleTable st = new StyleTable.Builder()
            .columnWidths(LengthUnit.Percent, 25, 75)
            .backgroundColor(Color.WHITE)
            .headerColor(Color.LIGHT_GRAY)
            .borderWidth(1)
            .whitespaceMode("pre-wrap")
            .build();

    private StyleChart sc = new StyleChart.Builder()
            .backgroundColor(Color.WHITE)
            .width(650,LengthUnit.Px)
            .height(350,LengthUnit.Px)
            .margin(LengthUnit.Px,65,50,50,10)
            .build();

    private StyleDiv sd = new StyleDiv.Builder()
            .margin(LengthUnit.Px,10,10,10,10)
            .width(100,LengthUnit.Percent)
            .build();

    public UIOptimizationRunnerStatusListener(ArbiterUIServer server) {
        this.server = server;
    }

    @Override
    public void onInitialization(IOptimizationRunner runner) {
        startTime = System.currentTimeMillis();

        OptimizationConfiguration conf = runner.getConfiguration();

        DataProvider<?> dataProvider = conf.getDataProvider();
        ScoreFunction<?,?> scoreFunction = conf.getScoreFunction();
        ResultSaver<?,?,?> resultSaver = conf.getResultSaver();
        ParameterSpace<?> space = conf.getCandidateGenerator().getParameterSpace();

        String[][] table = new String[][]{
                {"Candidate Generator:",conf.getCandidateGenerator().toString()},
                {"Data Provider:",(dataProvider == null ? "" : dataProvider.toString())},
                {"Score Function:", (scoreFunction == null ? "" : scoreFunction.toString())},
                {"Result Saver:", (resultSaver == null ? "" : resultSaver.toString())},
                {"Model Hyperparameter Space:", (space == null ? "" : space.toString())}
        };

        ComponentTable ct = new ComponentTable.Builder(st)
                .content(table)
                .build();

        server.updateOptimizationSettings(ct);
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        doSummaryStatusUpdate(runner);
    }

    @Override
    public void onStatusChange(IOptimizationRunner runner) {
        doSummaryStatusUpdate(runner);
    }

    private void doSummaryStatusUpdate(IOptimizationRunner<?,?,?> runner){
        long currentTime = System.currentTimeMillis();
        Double bestScore = runner.bestScore();
        int bestScoreIdx = runner.bestScoreCandidateIndex();
        Long scoreTime = runner.bestScoreTime();
        long durationSinceBest = (scoreTime == null ? 0 : currentTime - scoreTime);

        int completed = runner.numCandidatesCompleted();
        int queued = runner.numCandidatesQueued();
        int failed = runner.numCandidatesFailed();
        int total = runner.numCandidatesTotal();

        long totalRuntime = currentTime - startTime;

        String[][] table = new String[][]{
                {"Configurations Completed:",String.valueOf(completed)},
                {"Configurations Queued/Running:",String.valueOf(queued)},
                {"Configurations Failed:", String.valueOf(failed)},
                {"Configurations Total:", String.valueOf(total)},
                {"Best Score:", (bestScoreIdx == -1 ? "-" : String.valueOf(bestScore)) },
                {"Best Score Model Index:", (bestScoreIdx == -1 ? "-" : String.valueOf(bestScoreIdx)) },
                {"Best Score Model Found At:",
                        (bestScoreIdx == -1 ? "-" : formatTimeMS(scoreTime) + " (" + formatDurationMS(durationSinceBest,true) + " ago)") },
                {"Total Runtime:",formatDurationMS(totalRuntime,true)},
                {"Termination Conditions:",runner.getConfiguration().getTerminationConditions().toString()}
        };

        List<Component> components = new ArrayList<>();

        Component tableC = new ComponentTable.Builder(st)
                .content(table)
                .build();
        components.add(tableC);




        if(bestScoreIdx >= 0){
            //Actually have at least one candidate with a score...
            if(lastBestScoreTime == -1 ){
                //First candidate:
                lastBestScore = bestScore;
                lastBestScoreTime = currentTime;
            } else if(bestScore != lastBestScore){
                //New/improved candidate:
                bestScores.add(bestScore);
                bestScoreStartTimes.add(scoreTime);
                lastBestScoreTime = currentTime;
                lastBestScore = bestScore;
            }

            int nScores = bestScores.size();
            //Produce graph. Here: Want a step type graph
            double[] scores = new double[2*nScores];
            double[] times = new double[2*nScores];
            for( int i=0; i<nScores; i++ ){
                scores[2*i] = bestScores.get(i);
                scores[2*i+1] = scores[2*i];
                times[2*i] = (bestScoreStartTimes.get(i) - startTime) / 60000.0; //convert to minutes since start
                if(i+1<nScores) times[2*i+1] = (bestScoreStartTimes.get(i+1) - startTime) / 60000.0;
            }
            //Last point: current time
            scores[2*nScores-1] = lastBestScore;
            times[2*nScores-1] = (currentTime-startTime) / 60000.0;

            ChartLine chart = new ChartLine.Builder("Best Model Score vs. Time (mins)", sc)
                    .addSeries("Score vs. Time (mins)",times,scores)
                    .build();

            components.add(chart);
        }

        //Create a plot of all models vs. time
        List<CandidateStatus> statusList = runner.getCandidateStatus();
        List<CandidateStatus> completedStatuses = new ArrayList<>();
        for(CandidateStatus s : statusList){
            if(s.getStatus() == Status.Complete) completedStatuses.add(s);
        }
        Collections.sort(completedStatuses, new Comparator<CandidateStatus>() {
            @Override
            public int compare(CandidateStatus o1, CandidateStatus o2) {
                return Long.compare(o1.getEndTime(),o2.getEndTime());
            }
        });

        double[] time = new double[completedStatuses.size()];
        double[] score = new double[completedStatuses.size()];
        for( int i=0; i<completedStatuses.size(); i++ ){
            CandidateStatus cs = completedStatuses.get(i);
            time[i] = (cs.getEndTime() - startTime) / 60000.0;  //minutes since start
            Double temp = cs.getScore();
            score[i] = (temp == null ? Double.NaN : temp);
        }

        ChartScatter allCandidateScores = new ChartScatter.Builder("All Candidate Scores",sc)
                .addSeries("Score vs. Time (mins)",time,score)
                .build();
        components.add(allCandidateScores);

        ComponentDiv cd = new ComponentDiv(sd, components.toArray(new Component[components.size()]));
        server.updateStatus(cd);

        server.updateResults(runner.getCandidateStatus());
    }

    /** Convert timestamp to String */
    private String formatTimeMS(Long time){
        if(time == null) return "null";
        Calendar c = Calendar.getInstance(TimeZone.getDefault());
        c.setTimeInMillis(time);
        int min = c.get(Calendar.MINUTE);
        return c.get(Calendar.HOUR_OF_DAY) + ":" + (min <= 9 ? "0" : "") + min;
    }

    /** Convert duration (in ms) to format such as "1hr 24min" */
    private String formatDurationMS(long durationMS, boolean filterNegative){
        if(filterNegative && durationMS <= 0) return "0 min";
        long hours = (durationMS / 3600000);
        long partHour = durationMS % 3600000;
        long min = partHour / 60000;
        if(hours > 0) return hours + " hr, " + min + " min";
        else return min + " min";
    }

    @Override
    public void onCompletion(OptimizationResult<?, ?, ?> result) {

    }
}
