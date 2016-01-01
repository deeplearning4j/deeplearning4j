package org.arbiter.optimize.ui.listener;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor @Data
public class SummaryStatus {

    public SummaryStatus(){
        //No arg constructor required for jersey/jackson
    }

    private long currentTime;
    private double bestScore;
    private long bestScoreTime;
    private int numCompleted;
    private int numQueued;
    private int numFailed;
    private int numTotal;

}
