package org.deeplearning4j.nearestneighbor.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.clustering.vptree.VPTreeFillSearch;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nearestneighbor.model.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.serde.binary.BinarySerde;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.util.*;

import static play.mvc.Controller.request;
import static play.mvc.Results.*;

/**
 * A rest server for using an
 * {@link VPTree} based on loading an ndarray containing
 * the data points for the path
 * The input values are an {@link CSVRecord}
 * which (based on the input schema) will automatically
 * have their values transformed.
 *
 * @author Adam Gibson
 */
@Slf4j
public class NearestNeighborsServer {
    @Parameter(names = {"--ndarrayPath"}, arity = 1, required = true)
    private String ndarrayPath = null;
    @Parameter(names = {"--labelsPath"}, arity = 1, required = false)
    private String labelsPath = null;
    @Parameter(names = {"--nearestNeighborsPort"}, arity = 1)
    private int port = 9000;
    @Parameter(names = {"--similarityFunction"}, arity = 1)
    private String similarityFunction = "euclidean";
    @Parameter(names = {"--invert"}, arity = 1)
    private boolean invert = false;

    private Server server;

    public void runMain(String... args) throws Exception {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            log.error("Error in NearestNeighboursServer parameters", e);
            StringBuilder sb = new StringBuilder();
            jcmdr.usage(sb);
            log.error("Usage: {}", sb.toString());

            //User provides invalid input -> print the usage info
            jcmdr.usage();
            if (ndarrayPath == null)
                log.error("Json path parameter is missing (null)");
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        try {
            runHelper();
        } catch (Throwable t){
            log.error("Error in NearestNeighboursServer run method",t);
        }
    }

    protected void runHelper() throws Exception {

        String[] pathArr = ndarrayPath.split(",");
        //INDArray[] pointsArr = new INDArray[pathArr.length];
        // first of all we reading shapes of saved eariler files
        int rows = 0;
        int cols = 0;
        for (int i = 0; i < pathArr.length; i++) {
            DataBuffer shape = BinarySerde.readShapeFromDisk(new File(pathArr[i]));

            log.info("Loading shape {} of {}; Shape: [{} x {}]", i + 1, pathArr.length, Shape.size(shape, 0),
                            Shape.size(shape, 1));

            if (Shape.rank(shape) != 2)
                throw new DL4JInvalidInputException("NearestNeighborsServer assumes 2D chunks");

            rows += Shape.size(shape, 0);

            if (cols == 0)
                cols = Shape.size(shape, 1);
            else if (cols != Shape.size(shape, 1))
                throw new DL4JInvalidInputException(
                                "NearestNeighborsServer requires equal 2D chunks. Got columns mismatch.");
        }

        final List<String> labels = new ArrayList<>();
        if (labelsPath != null) {
            String[] labelsPathArr = labelsPath.split(",");
            for (int i = 0; i < labelsPathArr.length; i++) {
                labels.addAll(FileUtils.readLines(new File(labelsPathArr[i]), "utf-8"));
            }
        }
        if (!labels.isEmpty() && labels.size() != rows)
            throw new DL4JInvalidInputException(String.format("Number of labels must match number of rows in points matrix (expected %d, found %d)", rows, labels.size()));

        final INDArray points = Nd4j.createUninitialized(rows, cols);

        int lastPosition = 0;
        for (int i = 0; i < pathArr.length; i++) {
            log.info("Loading chunk {} of {}", i + 1, pathArr.length);
            INDArray pointsArr = BinarySerde.readFromDisk(new File(pathArr[i]));

            points.get(NDArrayIndex.interval(lastPosition, lastPosition + pointsArr.rows())).assign(pointsArr);
            lastPosition += pointsArr.rows();

            // let's ensure we don't bring too much stuff in next loop
            System.gc();
        }

        VPTree tree = new VPTree(points, similarityFunction, invert);

        RoutingDsl routingDsl = new RoutingDsl();
        //return the host information for a given id
        routingDsl.POST("/knn").routeTo(FunctionUtil.function0((() -> {
            try {
                NearestNeighborRequest record = Json.fromJson(request().body().asJson(), NearestNeighborRequest.class);
                NearestNeighbor nearestNeighbor =
                                NearestNeighbor.builder().points(points).record(record).tree(tree).build();

                if (record == null)
                    return badRequest(Json.toJson(Collections.singletonMap("status", "invalid json passed.")));

                NearstNeighborsResults results =
                                NearstNeighborsResults.builder().results(nearestNeighbor.search()).build();


                return ok(Json.toJson(results));

            } catch (Throwable e) {
                log.error("Error in POST /knn",e);
                e.printStackTrace();
                return internalServerError(e.getMessage());
            }
        })));

        routingDsl.POST("/knnnew").routeTo(FunctionUtil.function0((() -> {
            try {
                Base64NDArrayBody record = Json.fromJson(request().body().asJson(), Base64NDArrayBody.class);
                if (record == null)
                    return badRequest(Json.toJson(Collections.singletonMap("status", "invalid json passed.")));

                INDArray arr = Nd4jBase64.fromBase64(record.getNdarray());
                List<DataPoint> results;
                List<Double> distances;

                if (record.isForceFillK()) {
                    VPTreeFillSearch vpTreeFillSearch = new VPTreeFillSearch(tree, record.getK(), arr);
                    vpTreeFillSearch.search();
                    results = vpTreeFillSearch.getResults();
                    distances = vpTreeFillSearch.getDistances();
                } else {
                    results = new ArrayList<>();
                    distances = new ArrayList<>();
                    tree.search(arr, record.getK(), results, distances);
                }

                if (results.size() != distances.size()) {
                    return internalServerError(
                            String.format("results.size == %d != %d == distances.size",
                                    results.size(), distances.size()));
                }

                List<NearestNeighborsResult> nnResult = new ArrayList<>();
                for (int i=0; i<results.size(); i++) {
                    if (!labels.isEmpty())
                        nnResult.add(new NearestNeighborsResult(results.get(i).getIndex(), distances.get(i), labels.get(results.get(i).getIndex())));
                    else
                        nnResult.add(new NearestNeighborsResult(results.get(i).getIndex(), distances.get(i)));
                }

                NearestNeighborsResults results2 = NearestNeighborsResults.builder().results(nnResult).build();
                return ok(Json.toJson(results2));

            } catch (Throwable e) {
                log.error("Error in POST /knnnew",e);
                e.printStackTrace();
                return internalServerError(e.getMessage());
            }
        })));

        //Set play secret key, if required
        //http://www.playframework.com/documentation/latest/ApplicationSecret
        String crypto = System.getProperty("play.crypto.secret");
        if (crypto == null || "changeme".equals(crypto) || "".equals(crypto)) {
            byte[] newCrypto = new byte[1024];

            new Random().nextBytes(newCrypto);

            String base64 = Base64.getEncoder().encodeToString(newCrypto);
            System.setProperty("play.crypto.secret", base64);
        }

        server = Server.forRouter(routingDsl.build(), Mode.PROD, port);


    }

    /**
     * Stop the server
     */
    public void stop() {
        if (server != null) {
            log.info("Attempting to stop server");
            server.stop();
        }
    }

    public static void main(String[] args) throws Exception {
        new NearestNeighborsServer().runMain(args);
    }

}
