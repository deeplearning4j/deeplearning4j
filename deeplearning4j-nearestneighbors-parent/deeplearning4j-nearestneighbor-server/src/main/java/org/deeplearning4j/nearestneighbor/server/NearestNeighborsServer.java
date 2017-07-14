package org.deeplearning4j.nearestneighbor.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.clustering.vptree.VPTreeFillSearch;
import org.deeplearning4j.nearestneighbor.model.*;
import org.jboss.netty.util.internal.ByteBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.serde.binary.BinarySerde;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static play.mvc.Controller.request;
import static play.mvc.Results.badRequest;
import static play.mvc.Results.internalServerError;
import static play.mvc.Results.ok;

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
public class NearestNeighborsServer {
    @Parameter(names = {"--ndarrayPath"}, arity = 1, required = true)
    private String ndarrayPath = null;
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
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            if (ndarrayPath == null)
                System.err.println("Json path parameter is missing.");
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        String[] pathArr = ndarrayPath.split(",");
        INDArray[] pointsArr = new INDArray[pathArr.length];
        for (int i = 0; i < pathArr.length; i++)
            pointsArr[i] = BinarySerde.readFromDisk(new File(pathArr[i]));
        final INDArray points = Nd4j.concat(0, pointsArr);
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

            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
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

                List<NearestNeighborsResult> nnResult = new ArrayList<>();
                for (DataPoint dataPoint : results) {
                    nnResult.add(new NearestNeighborsResult(dataPoint.getIndex()));
                }

                NearstNeighborsResults results2 = NearstNeighborsResults.builder().results(nnResult).build();
                return ok(Json.toJson(results2));

            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        server = Server.forRouter(routingDsl.build(), Mode.DEV, port);


    }

    /**
     * Stop the server
     */
    public void stop() {
        if (server != null)
            server.stop();
    }

    public static void main(String[] args) throws Exception {
        new NearestNeighborsServer().runMain(args);
    }

}
