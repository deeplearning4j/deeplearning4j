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

//    public long getCurrentTime() {
//        return this.currentTime;
//    }
//
//    public double getBestScore() {
//        return this.bestScore;
//    }
//
//    public long getBestScoreTime() {
//        return this.bestScoreTime;
//    }
//
//    public int getNumCompleted() {
//        return this.numCompleted;
//    }
//
//    public int getNumQueued() {
//        return this.numQueued;
//    }
//
//    public int getNumFailed() {
//        return this.numFailed;
//    }
//
//    public void setCurrentTime(long currentTime) {
//        this.currentTime = currentTime;
//    }
//
//    public void setBestScore(double bestScore) {
//        this.bestScore = bestScore;
//    }
//
//    public void setBestScoreTime(long bestScoreTime) {
//        this.bestScoreTime = bestScoreTime;
//    }
//
//    public void setNumCompleted(int numCompleted) {
//        this.numCompleted = numCompleted;
//    }
//
//    public void setNumQueued(int numQueued) {
//        this.numQueued = numQueued;
//    }
//
//    public void setNumFailed(int numFailed) {
//        this.numFailed = numFailed;
//    }
//
//    public boolean equals(Object o) {
//        if (o == this) return true;
//        if (!(o instanceof SummaryStatus)) return false;
//        final SummaryStatus other = (SummaryStatus) o;
//        if (!other.canEqual((Object) this)) return false;
//        if (this.currentTime != other.currentTime) return false;
//        if (Double.compare(this.bestScore, other.bestScore) != 0) return false;
//        if (this.bestScoreTime != other.bestScoreTime) return false;
//        if (this.numCompleted != other.numCompleted) return false;
//        if (this.numQueued != other.numQueued) return false;
//        if (this.numFailed != other.numFailed) return false;
//        return true;
//    }
//
//    public int hashCode() {
//        final int PRIME = 59;
//        int result = 1;
//        final long $currentTime = this.currentTime;
//        result = result * PRIME + (int) ($currentTime >>> 32 ^ $currentTime);
//        final long $bestScore = Double.doubleToLongBits(this.bestScore);
//        result = result * PRIME + (int) ($bestScore >>> 32 ^ $bestScore);
//        final long $bestScoreTime = this.bestScoreTime;
//        result = result * PRIME + (int) ($bestScoreTime >>> 32 ^ $bestScoreTime);
//        result = result * PRIME + this.numCompleted;
//        result = result * PRIME + this.numQueued;
//        result = result * PRIME + this.numFailed;
//        return result;
//    }
//
//    protected boolean canEqual(Object other) {
//        return other instanceof SummaryStatus;
//    }
//
//    public String toString() {
//        return "org.arbiter.optimize.ui.listener.SummaryStatus(currentTime=" + this.currentTime + ", bestScore=" + this.bestScore + ", bestScoreTime=" + this.bestScoreTime + ", numCompleted=" + this.numCompleted + ", numQueued=" + this.numQueued + ", numFailed=" + this.numFailed + ")";
//    }
}
