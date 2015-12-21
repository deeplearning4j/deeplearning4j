package org.arbiter.optimize.ui.listener;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.runner.IOptimizationRunner;
import org.arbiter.optimize.runner.listener.StatusListener;
import org.arbiter.optimize.ui.ArbiterUIServer;

@AllArgsConstructor
public class UIStatusListener implements StatusListener {

    private ArbiterUIServer server;

    @Override
    public void onInitialization(IOptimizationRunner runner) {

    }

    @Override
    public void onShutdown(IOptimizationRunner runner) {

    }

    @Override
    public void onStatusChange(IOptimizationRunner runner) {
        long currentTime = System.currentTimeMillis();
        double score = runner.bestScore();
        long scoreTime = runner.bestScoreTime();
        int queued = runner.numCandidatesScheduled();
        int completed = runner.numCandidatesCompleted();
        int failed = runner.numCandidatesFailed();
        SummaryStatus status = new SummaryStatus(currentTime,score,scoreTime,
                completed,queued,failed);
        server.updateStatus(status);
    }

    @Override
    public void onCompletion(OptimizationResult<?, ?, ?> result) {

    }
}
