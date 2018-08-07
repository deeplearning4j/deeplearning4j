/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.gym;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *
 * ClientUtils contain the utility methods to post and get data from the server REST API through the library unirest.
 */
public class ClientUtils {

    static public JsonNode post(String url, JSONObject json) {
        HttpResponse<JsonNode> jsonResponse = null;

        try {
            jsonResponse = Unirest.post(url).header("content-type", "application/json").body(json).asJson();
        } catch (UnirestException e) {
            unirestCrash(e);
        }

        return jsonResponse.getBody();
    }


    static public JSONObject get(String url) {
        HttpResponse<JsonNode> jsonResponse = null;

        try {
            jsonResponse = Unirest.get(url).header("content-type", "application/json").asJson();
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
            throw new RuntimeException("Couldn't parse json reply.", e);
        else
            throw new RuntimeException("Connection error", e);
    }


}
