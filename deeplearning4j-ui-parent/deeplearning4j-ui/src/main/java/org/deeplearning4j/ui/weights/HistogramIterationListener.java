package org.deeplearning4j.ui.weights;


import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.deeplearning4j.ui.UiUtils;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.stats.api.StatsUpdateConfiguration;
import org.deeplearning4j.ui.stats.impl.DefaultStatsUpdateConfiguration;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;

/**
 *
 * A histogram iteration listener that updates the weights of the model with a web based ui.
 *
 * @deprecated Use {@link StatsListener} and {@link UIServer#attach(StatsStorage)}. See examples repo for how.
 *
 * @author Adam Gibson
 */
@Deprecated
@Slf4j
public class HistogramIterationListener extends StatsListener {
    private boolean openBrowser;
    private boolean firstIteration = true;
    private String path;
    private static final String subPath = "weights";

    public HistogramIterationListener(@NonNull UiConnectionInfo connection, int iterations) {
        this(new InMemoryStatsStorage(), iterations, true);

    }

    public HistogramIterationListener(int iterations) {
        this(iterations, true);
    }

    public HistogramIterationListener(int iterations, boolean openBrowser) {
        this(new InMemoryStatsStorage(), iterations, openBrowser);
    }

    public HistogramIterationListener(StatsStorage ssr, int iterations, boolean openBrowser) {
        super(ssr, null, getUpdateConfiguration(iterations), null, null);
        int port = -1;
        try {
            UIServer server = UIServer.getInstance();
            port = server.getPort();
        } catch (Exception e) {
            log.error("Error initializing UI server", e);
            throw new RuntimeException(e);
        }

        UIServer.getInstance().attach(ssr);

        this.path = "http://localhost:" + port + "/" + subPath;
        this.openBrowser = openBrowser;

        System.out.println("UI Histogram URL: " + this.path);
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        super.iterationDone(model, iteration);

        if (openBrowser && firstIteration) {
            StringBuilder builder =
                            new StringBuilder("http://localhost:").append(UIServer.getInstance().getPort()).append("/"); ///connectionInfo.getFullAddress());
            builder.append(subPath).append("?sid=").append(super.getSessionID());
            UiUtils.tryOpenBrowser(builder.toString(), log);
            firstIteration = false;
        }
    }


    private static StatsUpdateConfiguration getUpdateConfiguration(int iterations) {
        //Note: we don't *need* all of these stats just for histogram listener - but other info
        // is still available at /train
        return new DefaultStatsUpdateConfiguration.Builder().reportingFrequency(iterations).build();
    }
}
