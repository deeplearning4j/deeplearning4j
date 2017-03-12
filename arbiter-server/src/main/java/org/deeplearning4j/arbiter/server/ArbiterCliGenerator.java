package org.deeplearning4j.arbiter.server;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.optimize.candidategenerator.GridSearchCandidateGenerator;

import java.io.File;

/**
 * Created by agibsonccc on 3/12/17.
 */
public class ArbiterCliGenerator {

    private String searchSpacePath = null;
    private String candidateType = null;
    private int discretizationCount = 5;
    private String gridSearchOrder = null;


    public final static String RANDOM_CANDIDATE = "random";
    public final static String GRID_SEARCH_CANDIDATE = "gridsearch";

    public final static String SEQUENTIAL_ORDER = "sequence";
    public final static String RANDOM_ORDER = "random";


    public void runMain(String...args) throws Exception  {

    }

    public static void main(String...args) throws Exception {
        new ArbiterCliGenerator().runMain(args);
    }

    private GridSearchCandidateGenerator.Mode getMode() {
        if(gridSearchOrder.equals(RANDOM_ORDER))
            return GridSearchCandidateGenerator.Mode.RandomOrder;
        else if(gridSearchOrder.equals(SEQUENTIAL_ORDER)) {
            return GridSearchCandidateGenerator.Mode.Sequential;
        }
        else throw new IllegalArgumentException("Illegal mode " + gridSearchOrder);
    }

    private MultiLayerSpace loadMultiLayer() throws Exception {
        MultiLayerSpace multiLayerSpace = MultiLayerSpace.fromJson(FileUtils.readFileToString(new File(searchSpacePath)));
        return multiLayerSpace;
    }
}
