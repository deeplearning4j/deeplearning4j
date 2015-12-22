package org.arbiter.optimize.ui.listener;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.config.OptimizationConfiguration;
import org.arbiter.optimize.runner.IOptimizationRunner;
import org.arbiter.optimize.runner.listener.StatusListener;
import org.arbiter.optimize.ui.ArbiterUIServer;

@AllArgsConstructor
public class UIStatusListener implements StatusListener {

    private ArbiterUIServer server;

    @Override
    public void onInitialization(IOptimizationRunner runner) {
        //TODO do this better
        OptimizationConfiguration conf = runner.getConfiguration();

        StringBuilder sb = new StringBuilder();
        sb.append("Candidate generator: ").append(conf.getCandidateGenerator()).append("\n")
            .append("Data Provider: ").append(conf.getDataProvider()).append("\n")
            .append("Score Function: ").append(conf.getScoreFunction()).append("\n")
            .append("Result saver: ").append(conf.getResultSaver()).append("\n")
            .append("Model hyperparameter space: ").append(conf.getCandidateGenerator().getParameterSpace());

        server.updateOptimizationSettings(sb.toString());
    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {

    }

    @Override
    public void onStatusChange(IOptimizationRunner runner) {
        long currentTime = System.currentTimeMillis();
        double score = runner.bestScore();
        long scoreTime = runner.bestScoreTime();

        int completed = runner.numCandidatesCompleted();
        int queued = runner.numCandidatesQueued();
        int failed = runner.numCandidatesFailed();
        int total = runner.numCandidatesTotal();

        SummaryStatus status = new SummaryStatus(currentTime,score,scoreTime,
                completed,queued,failed,total);
        server.updateStatus(status);

        server.updateResults(runner.getCandidateStatus());
    }

    @Override
    public void onCompletion(OptimizationResult<?, ?, ?> result) {

    }
}
