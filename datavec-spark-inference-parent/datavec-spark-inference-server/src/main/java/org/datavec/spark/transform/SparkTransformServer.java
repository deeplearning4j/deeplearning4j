package org.datavec.spark.transform;

import com.beust.jcommander.Parameter;
import com.fasterxml.jackson.databind.JsonNode;
import org.datavec.spark.transform.service.DataVecTransformService;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import play.server.Server;

import static play.mvc.Controller.request;

/**
 * Created by kepricon on 17. 6. 20.
 */
public abstract class SparkTransformServer implements DataVecTransformService {
    @Parameter(names = {"-j", "--jsonPath"}, arity = 1)
    protected String jsonPath = null;
    @Parameter(names = {"-dp", "--dataVecPort"}, arity = 1)
    protected int port = 9000;
    @Parameter(names = {"-dt", "--dataType"}, arity = 1)
    private TransformDataType transformDataType = null;
    protected Server server;
    protected static ObjectMapper objectMapper = new ObjectMapper();

    public abstract void runMain(String[] args) throws Exception;

    /**
     * Stop the server
     */
    public void stop() {
        if (server != null)
            server.stop();
    }

    protected String getJsonText() {
        JsonNode tryJson = request().body().asJson();
        if(tryJson != null)
            return tryJson.toString();
        else
            return request().body().asText();
    }
}
