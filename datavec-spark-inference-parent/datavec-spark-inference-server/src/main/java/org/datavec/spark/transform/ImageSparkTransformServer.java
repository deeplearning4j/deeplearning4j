package org.datavec.spark.transform;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.CSVRecord;
import org.datavec.spark.transform.model.ImageRecord;
import org.datavec.spark.transform.service.DataVecImageTransformService;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.io.IOException;

import static play.mvc.Controller.request;
import static play.mvc.Results.badRequest;
import static play.mvc.Results.internalServerError;
import static play.mvc.Results.ok;

/**
 * Created by kepricon on 17. 6. 19.
 */
@Slf4j
@Data
public class ImageSparkTransformServer implements DataVecImageTransformService{
    @Parameter(names = {"-j", "--jsonPath"}, arity = 1)
    private String jsonPath = null;
    @Parameter(names = {"-dp", "--dataVecPort"}, arity = 1)
    private int port = 9000;
    private Server server;
    private ImageSparkTransform transform;
    private static ObjectMapper objectMapper = new ObjectMapper();

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
            ImageTransformProcess transformProcess = ImageTransformProcess.fromJson(json);
            transform = new ImageSparkTransform(transformProcess);
        }
        else {
            log.warn("Server started with no json for transform process. Please ensure you specify a transform process via sending a post request with raw json" +
                    "to /transformprocess");
        }

        //return the host information for a given id
        routingDsl.GET("/transformprocess").routeTo(FunctionUtil.function0((() -> {
            try {
                if(transform == null)
                    return badRequest();
                log.info("Transform process initialized");
                return ok(Json.toJson(transform.getImageTransformProcess()));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transformprocess").routeTo(FunctionUtil.function0((() -> {
            try {
                ImageTransformProcess transformProcess = ImageTransformProcess.fromJson(getJsonText());
                setTransformProcess(transformProcess);
                log.info("Transform process initialized");
                return ok(Json.toJson(transformProcess));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transformincrementalarray").routeTo(FunctionUtil.function0((() -> {
            try {
                ImageRecord record = objectMapper.readValue(getJsonText(),ImageRecord.class);
                if (record == null)
                    return badRequest();
                return ok(Json.toJson(transformIncrementalArray(record)));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transformarray").routeTo(FunctionUtil.function0((() -> {
            try {
                BatchImageRecord batch = objectMapper.readValue(getJsonText(),BatchImageRecord.class);
                if (batch == null)
                    return badRequest();
                return ok(Json.toJson(transformArray(batch)));
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


    @Override
    public void setTransformProcess(ImageTransformProcess transformProcess) {
        this.transform = new ImageSparkTransform(transformProcess);
    }

    @Override
    public ImageTransformProcess transformProcess() {
        return transform.getImageTransformProcess();
    }

    @Override
    public Base64NDArrayBody transformIncrementalArray(ImageRecord record) throws IOException {
        return transform.toArray(record);
    }

    @Override
    public Base64NDArrayBody transformArray(BatchImageRecord batch) throws IOException {
        return transform.toArray(batch);
    }

    public static void main(String[] args) throws Exception {
        new ImageSparkTransformServer().runMain(args);
    }

    private String getJsonText() {
        JsonNode tryJson = request().body().asJson();
        if(tryJson != null)
            return tryJson.toString();
        else
            return request().body().asText();
    }
}
