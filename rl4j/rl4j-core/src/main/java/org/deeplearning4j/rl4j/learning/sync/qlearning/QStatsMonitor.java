package java.org.deeplearning4j.rl4j.learning.sync.qlearning;

import lombok.Getter;

public class QStatsMonitor {

    @Getter
    private Double startQ = Double.NaN;

    private int numQ = 0;
    private Double meanQ = 0D;

    public void add(Double maxQ) {
        if (!maxQ.isNaN()) {
            if (startQ.isNaN())
                startQ = maxQ;
            numQ++;
            meanQ += maxQ;
        }
    }

    public Double getMeanQ() {
        return meanQ / (numQ + 0.001); //avoid div zero
    }
}
