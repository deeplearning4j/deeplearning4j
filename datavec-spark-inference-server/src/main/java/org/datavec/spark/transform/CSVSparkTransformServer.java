package org.datavec.spark.transform;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.spark.transform.model.CSVRecord;
import play.Mode;
import play.libs.Json;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;

import static play.mvc.Controller.request;
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
public class CSVSparkTransformServer {
    @Parameter(names = {"-j","--jsonPath"},arity = 1,required = true)
    private String jsonPath = null;
    @Parameter(names = {"-dp","--dataVecPort"},arity = 1)
    private int port = 9000;

    public void runMain(String[] args) throws Exception {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch(ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            if(jsonPath == null)
                System.err.println("Json path parameter is missing.");
            try{ Thread.sleep(500); } catch(Exception e2){ }
            System.exit(1);
        }


        String json = FileUtils.readFileToString(new File(jsonPath));
        TransformProcess transformProcess = TransformProcess.fromJson(json);
        RoutingDsl routingDsl = new RoutingDsl();
        CSVSparkTransform transform = new CSVSparkTransform(transformProcess);
        //return the host information for a given id
        routingDsl.POST("/transform").routeTo(FunctionUtil.function0((() -> {
            try {
                return ok(Json.toJson(transform.transform(request().body().as(CSVRecord.class))));
            } catch (Exception e) {
                return internalServerError();
            }
        })));

        routingDsl.POST("/transformedarray").routeTo(FunctionUtil.function0((() -> {
            try {
                return ok(Json.toJson(transform.toArray(request().body().as(CSVRecord.class))));
            } catch (Exception e) {
                return internalServerError();
            }
        })));
        Server.forRouter( routingDsl.build(), Mode.DEV, port);


    }


    public static void main(String[] args) throws Exception {
        new CSVSparkTransformServer().runMain(args);
    }

}
