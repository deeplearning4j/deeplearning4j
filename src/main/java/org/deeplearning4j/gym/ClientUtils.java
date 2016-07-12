package org.deeplearning4j.gym;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * Created by rubenfiszel on 7/8/16.
 */
public class ClientUtils {


    static public JsonNode post(String url, JSONObject json) {
        HttpResponse<JsonNode> jsonResponse = null;

        try {
            jsonResponse = Unirest.post(url)
                    .header("content-type", "application/json")
                    .body(json)
                    .asJson();
        } catch (UnirestException e) {
            unirestCrash(e);
        }

        return jsonResponse.getBody();
    }


    static public JSONObject get(String url) {
        HttpResponse<JsonNode> jsonResponse = null;

        try {
            jsonResponse = Unirest.get(url)
                    .header("content-type", "application/json")
                    .asJson();
        } catch (UnirestException e) {
            unirestCrash(e);
        }

        checkReply(jsonResponse, url);

        return jsonResponse.getBody().getObject();
    }


    static public void checkReply(HttpResponse<JsonNode> res, String url) {
        if (res.getBody() == null)
            throw new RuntimeException("Invalid reply at: " + url);
    }

    static public void unirestCrash(UnirestException e) {
        //if couldn't parse json
        if (e.getCause().getCause().getCause() instanceof JSONException)
            throw new RuntimeException("Couldn't parse json reply. Probable cause: Wrong API Key at upload");
        else
            throw new RuntimeException("Connection error");
    }


}
