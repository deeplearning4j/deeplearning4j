package org.deeplearning4j.nearestneighbor.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.nearestneighbor.model.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.util.ArrayList;
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

    public void runMain(String[] args) throws Exception {
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

        String json = FileUtils.readFileToString(new File(ndarrayPath));
        INDArray points = Nd4jBase64.fromBase64(json);


        VPTree tree = new VPTree(points,similarityFunction,invert);


        RoutingDsl routingDsl = new RoutingDsl();
        //return the host information for a given id
        routingDsl.POST("/knn").routeTo(FunctionUtil.function0((() -> {
            try {
                NearestNeighborRequest record = Json.fromJson(request().body().asJson(), NearestNeighborRequest.class);
                NearestNeighbor nearestNeighbor = NearestNeighbor.builder()
                        .points(points).record(record).tree(tree).build();
                if (record == null)
                    return badRequest();
                NearstNeighborsResults results = NearstNeighborsResults.builder().results(nearestNeighbor.search()).build();
                return ok(Json.toJson(results));

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
