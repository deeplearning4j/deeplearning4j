package org.datavec.spark.transform;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.spark.transform.model.BatchRecord;
import org.datavec.spark.transform.model.CSVRecord;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.util.Collections;
import java.util.List;

import static play.mvc.Controller.request;
import static play.mvc.Results.badRequest;
import static play.mvc.Results.internalServerError;
import static play.mvc.Results.ok;

/**
 * A rest server for using an
 * {@link TransformProcess} based on simple
 * csv values and a schema via REST.
 *
 * The input values are an {@link CSVRecord}
 * which (based on the input schema) will automatically
 * have their values transformed.
 *
 * @author Adam Gibson
 */
@Slf4j
@Data
public class CSVSparkTransformServer {
    @Parameter(names = {"-j", "--jsonPath"}, arity = 1)
    private String jsonPath = null;
    @Parameter(names = {"-dp", "--dataVecPort"}, arity = 1)
    private int port = 9000;
    private Server server;
    private CSVSparkTransform transform;

    public void runMain(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            if (jsonPath == null)
                System.err.println("Json path parameter is missing.");
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }

        RoutingDsl routingDsl = new RoutingDsl();


       if(jsonPath != null) {
           String json = FileUtils.readFileToString(new File(jsonPath));
           TransformProcess transformProcess = TransformProcess.fromJson(json);
           transform = new CSVSparkTransform(transformProcess);
       }
       else {
           log.warn("Server started with no json for transform process. Please ensure you specify a transform process via sending a post request with raw json" +
                   "to /transformprocess");
       }

        //return the host information for a given id
        routingDsl.POST("/transformprocess").routeTo(FunctionUtil.function0((() -> {
            try {
                TransformProcess transformProcess = TransformProcess.fromJson(request().body().asJson().toString());
                transform = new CSVSparkTransform(transformProcess);
                log.info("Transform process initialized");
                return ok(Json.toJson(Collections.singletonMap("status","started")));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transformincremental").routeTo(FunctionUtil.function0((() -> {
            try {
                CSVRecord record = Json.fromJson(request().body().asJson(), CSVRecord.class);
                if (record == null)
                    return badRequest();
                return ok(Json.toJson(transform.transform(record)));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transform").routeTo(FunctionUtil.function0((() -> {
            try {
                BatchRecord batch = transform.transform(Json.fromJson(request().body().asJson(), BatchRecord.class));
                if (batch == null)
                    return badRequest();
                return ok(Json.toJson(batch));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        routingDsl.POST("/transformedincrementalarray").routeTo(FunctionUtil.function0((() -> {
            try {
                CSVRecord record = Json.fromJson(request().body().asJson(), CSVRecord.class);
                if (record == null)
                    return badRequest();
                return ok(Json.toJson(transform.toArray(record)));
            } catch (Exception e) {
                return internalServerError();
            }
        })));

        routingDsl.POST("/transformedarray").routeTo(FunctionUtil.function0((() -> {
            try {
                BatchRecord batchRecord = Json.fromJson(request().body().asJson(), BatchRecord.class);
                if (batchRecord == null)
                    return badRequest();
                return ok(Json.toJson(transform.toArray(batchRecord)));
            } catch (Exception e) {
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
        new CSVSparkTransformServer().runMain(args);
    }

}
