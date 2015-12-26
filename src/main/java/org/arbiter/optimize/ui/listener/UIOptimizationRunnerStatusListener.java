package org.arbiter.optimize.ui.listener;

import org.arbiter.optimize.api.ModelParameterSpace;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.saving.ResultSaver;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.runner.IOptimizationRunner;
import org.arbiter.optimize.runner.listener.runner.OptimizationRunnerStatusListener;
import org.arbiter.optimize.ui.ArbiterUIServer;
import org.arbiter.optimize.ui.components.RenderElements;
import org.arbiter.optimize.ui.components.RenderableComponent;
import org.arbiter.optimize.ui.components.RenderableComponentLineChart;
import org.arbiter.optimize.ui.components.RenderableComponentTable;

import java.util.*;

public class UIOptimizationRunnerStatusListener implements OptimizationRunnerStatusListener {

    private ArbiterUIServer server;
    private long startTime;
    private List<Double> bestScores = new ArrayList<>();
    private List<Long> bestScoreStartTimes = new ArrayList<>();
    private double lastBestScore;
    private long lastBestScoreTime = 0;

    public UIOptimizationRunnerStatusListener(ArbiterUIServer server) {
        this.server = server;
    }

    @Override
    public void onInitialization(IOptimizationRunner runner) {
        startTime = System.currentTimeMillis();

        OptimizationConfiguration conf = runner.getConfiguration();

        StringBuilder sb = new StringBuilder();
        sb.append("Candidate generator: ").append(conf.getCandidateGenerator()).append("\n")
            .append("Data Provider: ").append(conf.getDataProvider()).append("\n")
            .append("Score Function: ").append(conf.getScoreFunction()).append("\n")
            .append("Result saver: ").append(conf.getResultSaver()).append("\n")
            .append("Model hyperparameter space: ").append(conf.getCandidateGenerator().getParameterSpace());

        DataProvider<?> dataProvider = conf.getDataProvider();
        ScoreFunction<?,?> scoreFunction = conf.getScoreFunction();
        ResultSaver<?,?,?> resultSaver = conf.getResultSaver();
        ModelParameterSpace<?> space = conf.getCandidateGenerator().getParameterSpace();

        String[][] table = new String[][]{
                {"Candidate Generator:",conf.getCandidateGenerator().toString()},
                {"Data Provider:",(dataProvider == null ? "" : dataProvider.toString())},
                {"Score Function:", (scoreFunction == null ? "" : scoreFunction.toString())},
                {"Result Saver:", (resultSaver == null ? "" : resultSaver.toString())},
                {"Model Hyperparameter Space:", (space == null ? "" : space.toString())}
        };
        RenderElements elements = new RenderElements(new RenderableComponentTable(null,table));
        server.updateOptimizationSettings(elements);

//        server.updateOptimizationSettings(sb.toString());
        server.updateOptimizationSettings(elements);
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {
        doSummaryStatusUpdate(runner);
    }

    @Override
    public void onStatusChange(IOptimizationRunner runner) {
        doSummaryStatusUpdate(runner);
    }

    private void doSummaryStatusUpdate(IOptimizationRunner runner){
        long currentTime = System.currentTimeMillis();
        double bestScore = runner.bestScore();
        int bestScoreIdx = runner.bestScoreCandidateIndex();
        long scoreTime = runner.bestScoreTime();
        long durationSinceBest = currentTime - scoreTime;

        int completed = runner.numCandidatesCompleted();
        int queued = runner.numCandidatesQueued();
        int failed = runner.numCandidatesFailed();
        int total = runner.numCandidatesTotal();

        long totalRuntime = currentTime - startTime;

        String[][] table = new String[][]{
                {"Completed:",String.valueOf(completed)},
                {"Queued/Running:",String.valueOf(queued)},
                {"Failed:", String.valueOf(failed)},
                {"Total:", String.valueOf(total)},
                {"Best Score:", (bestScoreIdx == -1 ? "-" : String.valueOf(bestScore)) },
                {"Best Score Model Index:", (bestScoreIdx == -1 ? "-" : String.valueOf(bestScoreIdx)) },
                {"Best Score Model Found At:",
                        (bestScoreIdx == -1 ? "-" : formatTimeMS(scoreTime) + " (" + formatDurationMS(durationSinceBest,true) + " ago)") },
                {"Total Runtime:",formatDurationMS(totalRuntime,true)}
        };

        //TODO: best score vs. time
            //How? record this info manually...
            //Then chart as a step function
        //TODO: best score vs. iteration

        List<RenderableComponent> components = new ArrayList<>();
        components.add(new RenderableComponentTable(null,table));

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

            RenderableComponentLineChart chart = new RenderableComponentLineChart.Builder()
                    .addSeries("Score vs. Time (mins)",times,scores)
                    .title("Best model score vs. time")
                    .build();

            components.add(chart);

            System.out.println("SCORES VS TIME:");
            System.out.println("x: " + Arrays.toString(times));
            System.out.println("y: " + Arrays.toString(scores));
        }

        RenderElements elements = new RenderElements(components.toArray(new RenderableComponent[components.size()]));
        server.updateStatus(elements);

        server.updateResults(runner.getCandidateStatus());
    }

    /** Convert timestamp to String */
    private String formatTimeMS(long time){
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
