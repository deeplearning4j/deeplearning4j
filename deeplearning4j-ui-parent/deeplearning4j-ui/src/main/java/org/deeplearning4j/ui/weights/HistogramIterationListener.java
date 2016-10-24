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
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;

/**
 *
 * A histogram iteration listener that
 * updates the weights of the model
 * with a web based ui.
 *
 * @author Adam Gibson
 */
@Slf4j
public class HistogramIterationListener extends StatsListener {
//    private static final Logger log = LoggerFactory.getLogger(HistogramIterationListener.class);
//    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());
//    private WebTarget target;
//    private int iterations = 1;
//    private int curIteration = 0;
//    private ArrayList<Double> scoreHistory = new ArrayList<>();
//    private List<Map<String,List<Double>>> meanMagHistoryParams = new ArrayList<>();    //1 map per layer; keyed by new param name
//    private List<Map<String,List<Double>>> meanMagHistoryUpdates = new ArrayList<>();
//    private Map<String,Integer> layerNameIndexes = new HashMap<>();
//    private List<String> layerNames = new ArrayList<>();
//    private int layerNameIndexesCount = 0;
    private boolean openBrowser;
    private boolean firstIteration = true;
    private String path;
    private static final String subPath = "weights";
//    private UiConnectionInfo connectionInfo;

    public HistogramIterationListener(@NonNull UiConnectionInfo connection, int iterations) {
        this(new MapDBStatsStorage(), iterations, true);

    }

    public HistogramIterationListener(int iterations) {
        this(iterations, true);
    }

    public HistogramIterationListener(int iterations, boolean openBrowser) {
        this(new MapDBStatsStorage(), iterations, openBrowser);
    }

    public HistogramIterationListener(StatsStorage ssr, int iterations, boolean openBrowser){
        super(ssr, null, getUpdateConfiguration(iterations), null, null);
        int port = -1;
        try{
            UIServer server = UIServer.getInstance();
            port = server.getPort();
        }catch(Exception e){
            log.error("Error initializing UI server",e);
            throw new RuntimeException(e);
        }

        UIServer.getInstance().attach(ssr);

        this.path = "http://localhost:" + port + "/" + subPath;
        this.openBrowser = openBrowser;

        System.out.println("UI Histogram URL: " + this.path );
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        super.iterationDone(model, iteration);

        if(openBrowser && firstIteration){
            StringBuilder builder = new StringBuilder("http://localhost:")
                    .append(UIServer.getInstance().getPort()).append("/"); ///connectionInfo.getFullAddress());
            builder.append(subPath).append("?sid=").append(super.getSessionID());
            UiUtils.tryOpenBrowser(builder.toString(),log);
            firstIteration = false;
        }
    }


    private static StatsUpdateConfiguration getUpdateConfiguration(int iterations){
        return new DefaultStatsUpdateConfiguration.Builder()
                .reportingFrequency(iterations)
                .collectPerformanceStats(false)
                .collectMemoryStats(false)
                .collectGarbageCollectionStats(false)
                .collectLearningRates(false)

                .collectHistogramsParameters(true)
                .collectHistogramsUpdates(true)
                .collectHistogramsActivations(false)

                .collectMeanParameters(false)
                .collectMeanUpdates(false)
                .collectMeanActivations(false)

                .collectStdevParameters(false)
                .collectStdevUpdates(false)
                .collectStdevActivations(false)

                .collectMeanMagnitudesParameters(true)
                .collectMeanMagnitudesUpdates(true)
                .collectMeanMagnitudesActivations(false)
                .build();
    }
}
