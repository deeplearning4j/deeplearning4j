/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nearestneighbor.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.Vertx;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.vptree.VPTree;
import org.deeplearning4j.clustering.vptree.VPTreeFillSearch;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nearestneighbor.model.*;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.serde.binary.BinarySerde;

import java.io.File;
import java.util.*;

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
public class NearestNeighborsServer extends AbstractVerticle {

    private static class RunArgs {
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
    }

    private static RunArgs instanceArgs;
    private static NearestNeighborsServer instance;

    public NearestNeighborsServer(){ }

    public static NearestNeighborsServer getInstance(){
        return instance;
    }

    public static void runMain(String... args) {
        RunArgs r = new RunArgs();
        JCommander jcmdr = new JCommander(r);

        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            log.error("Error in NearestNeighboursServer parameters", e);
            StringBuilder sb = new StringBuilder();
            jcmdr.usage(sb);
            log.error("Usage: {}", sb.toString());

            //User provides invalid input -> print the usage info
            jcmdr.usage();
            if (r.ndarrayPath == null)
                log.error("Json path parameter is missing (null)");
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        instanceArgs = r;
        try {
            Vertx vertx = Vertx.vertx();
            vertx.deployVerticle(NearestNeighborsServer.class.getName());
        } catch (Throwable t){
            log.error("Error in NearestNeighboursServer run method",t);
        }
    }

    @Override
    public void start() throws Exception {
        instance = this;

        String[] pathArr = instanceArgs.ndarrayPath.split(",");
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
        if (instanceArgs.labelsPath != null) {
            String[] labelsPathArr = instanceArgs.labelsPath.split(",");
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

        VPTree tree = new VPTree(points, instanceArgs.similarityFunction, instanceArgs.invert);

        //Set play secret key, if required
        //http://www.playframework.com/documentation/latest/ApplicationSecret
        String crypto = System.getProperty("play.crypto.secret");
        if (crypto == null || "changeme".equals(crypto) || "".equals(crypto)) {
            byte[] newCrypto = new byte[1024];

            new Random().nextBytes(newCrypto);

            String base64 = Base64.getEncoder().encodeToString(newCrypto);
            System.setProperty("play.crypto.secret", base64);
        }

        Router r = Router.router(vertx);
        r.route().handler(BodyHandler.create());  //NOTE: Setting this is required to receive request body content at all
        createRoutes(r, labels, tree, points);

        vertx.createHttpServer()
                .requestHandler(r)
                .listen(instanceArgs.port);
    }

    private void createRoutes(Router r, List<String> labels, VPTree tree, INDArray points){

        r.post("/knn").handler(rc -> {
            try {
                String json = rc.getBodyAsJson().encode();
                NearestNeighborRequest record = JsonMappers.getMapper().readValue(json, NearestNeighborRequest.class);

                NearestNeighbor nearestNeighbor =
                        NearestNeighbor.builder().points(points).record(record).tree(tree).build();

                if (record == null) {
                    rc.response().setStatusCode(HttpResponseStatus.BAD_REQUEST.code())
                            .putHeader("content-type", "application/json")
                            .end(JsonMappers.getMapper().writeValueAsString(Collections.singletonMap("status", "invalid json passed.")));
                    return;
                }

                NearestNeighborsResults results = NearestNeighborsResults.builder().results(nearestNeighbor.search()).build();

                rc.response().setStatusCode(HttpResponseStatus.BAD_REQUEST.code())
                        .putHeader("content-type", "application/json")
                        .end(JsonMappers.getMapper().writeValueAsString(results));
                return;
            } catch (Throwable e) {
                log.error("Error in POST /knn",e);
                rc.response().setStatusCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code())
                        .end("Error parsing request - " + e.getMessage());
                return;
            }
        });

        r.post("/knnnew").handler(rc -> {
            try {
                String json = rc.getBodyAsJson().encode();
                Base64NDArrayBody record = JsonMappers.getMapper().readValue(json, Base64NDArrayBody.class);
                if (record == null) {
                    rc.response().setStatusCode(HttpResponseStatus.BAD_REQUEST.code())
                            .putHeader("content-type", "application/json")
                            .end(JsonMappers.getMapper().writeValueAsString(Collections.singletonMap("status", "invalid json passed.")));
                    return;
                }

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
                    rc.response()
                            .setStatusCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code())
                            .end(String.format("results.size == %d != %d == distances.size", results.size(), distances.size()));
                    return;
                }

                List<NearestNeighborsResult> nnResult = new ArrayList<>();
                for (int i=0; i<results.size(); i++) {
                    if (!labels.isEmpty())
                        nnResult.add(new NearestNeighborsResult(results.get(i).getIndex(), distances.get(i), labels.get(results.get(i).getIndex())));
                    else
                        nnResult.add(new NearestNeighborsResult(results.get(i).getIndex(), distances.get(i)));
                }

                NearestNeighborsResults results2 = NearestNeighborsResults.builder().results(nnResult).build();
                String j = JsonMappers.getMapper().writeValueAsString(results2);
                rc.response()
                        .putHeader("content-type", "application/json")
                        .end(j);
            } catch (Throwable e) {
                log.error("Error in POST /knnnew",e);
                rc.response().setStatusCode(HttpResponseStatus.INTERNAL_SERVER_ERROR.code())
                        .end("Error parsing request - " + e.getMessage());
                return;
            }
        });
    }

    /**
     * Stop the server
     */
    public void stop() throws Exception {
        super.stop();
    }

    public static void main(String[] args) throws Exception {
        runMain(args);
    }

}
