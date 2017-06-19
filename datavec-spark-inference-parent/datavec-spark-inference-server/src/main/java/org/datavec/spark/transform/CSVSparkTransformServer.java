package org.datavec.spark.transform;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchRecord;
import org.datavec.spark.transform.model.CSVRecord;
import org.datavec.spark.transform.service.DataVecTransformService;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.io.IOException;

import static play.mvc.Controller.request;
import static play.mvc.Results.*;

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
public class CSVSparkTransformServer implements DataVecTransformService {
    @Parameter(names = {"-j", "--jsonPath"}, arity = 1)
    private String jsonPath = null;
    @Parameter(names = {"-dp", "--dataVecPort"}, arity = 1)
    private int port = 9000;
    private Server server;
    private CSVSparkTransform transform;
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
            TransformProcess transformProcess = TransformProcess.fromJson(json);
            transform = new CSVSparkTransform(transformProcess);
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
                return ok(Json.toJson(transform.getTransformProcess()));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transformprocess").routeTo(FunctionUtil.function0((() -> {
            try {
                TransformProcess transformProcess = TransformProcess.fromJson(request().body().asJson().toString());
                setTransformProcess(transformProcess);
                log.info("Transform process initialized");
                return ok(Json.toJson(transformProcess));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transformincremental").routeTo(FunctionUtil.function0((() -> {
            try {
                CSVRecord record = objectMapper.readValue(request().body().asText(),CSVRecord.class);
                if (record == null)
                    return badRequest();
                return ok(Json.toJson(transformIncremental(record)));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        //return the host information for a given id
        routingDsl.POST("/transform").routeTo(FunctionUtil.function0((() -> {
            try {
                BatchRecord batch = transform(objectMapper.readValue(request().body().asText(),BatchRecord.class));
                if (batch == null)
                    return badRequest();
                return ok(Json.toJson(batch));
            } catch (Exception e) {
                e.printStackTrace();
                return internalServerError();
            }
        })));

        routingDsl.POST("/transformincrementalarray").routeTo(FunctionUtil.function0((() -> {
            try {
                CSVRecord record =  objectMapper.readValue(request().body().asText(),CSVRecord.class);
                if (record == null)
                    return badRequest();
                return ok(Json.toJson(transformArrayIncremental(record)));
            } catch (Exception e) {
                return internalServerError();
            }
        })));

        routingDsl.POST("/transformarray").routeTo(FunctionUtil.function0((() -> {
            try {
                BatchRecord batchRecord =  objectMapper.readValue(request().body().asText(),BatchRecord.class);
                if (batchRecord == null)
                    return badRequest();
                return ok(Json.toJson(transformArray(batchRecord)));
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

    /**
     * @param transformProcess
     */
    @Override
    public void setTransformProcess(TransformProcess transformProcess) {
        this.transform = new CSVSparkTransform(transformProcess);
    }

    /**
     * @return
     */
    @Override
    public TransformProcess transformProcess() {
        return transform.getTransformProcess();
    }

    /**
     * @param transform
     * @return
     */
    @Override
    public CSVRecord transformIncremental(CSVRecord transform) {
        return this.transform.transform(transform);
    }

    /**
     * @param batchRecord
     * @return
     */
    @Override
    public BatchRecord transform(BatchRecord batchRecord) {
        return transform.transform(batchRecord);
    }

    /**
     * @param batchRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArray(BatchRecord batchRecord) {
        try {
            return this.transform.toArray(batchRecord);
        } catch (IOException e) {
           throw new IllegalStateException("Transform array shouldn't throw exception");
        }
    }

    /**
     * @param csvRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArrayIncremental(CSVRecord csvRecord) {
        try {
            return this.transform.toArray(csvRecord);
        } catch (IOException e) {
            throw new IllegalStateException("Transform array shouldn't throw exception");
        }    }
}
