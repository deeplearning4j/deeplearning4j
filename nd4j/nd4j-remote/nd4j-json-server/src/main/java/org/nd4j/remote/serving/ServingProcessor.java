package org.nd4j.remote.serving;

public class ServingProcessor {

    public String listEndpoints() {
        String retVal = "/v1/ \n/v1/serving/";
        return retVal;
    }

    public String processModel(String body) {
        String response = null; //"Not implemented";
        return response;
    }
}
