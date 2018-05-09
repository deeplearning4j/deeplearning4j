package org.deeplearning4j.nearestneighbor.client;

import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.request.HttpRequest;
import com.mashape.unirest.request.HttpRequestWithBody;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nearestneighbor.model.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.serde.base64.Nd4jBase64;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;

/**
 * Client for the nearest neighbors server.
 *  To create a client, pass in a host port combination with the following format:
 *  http://host:port
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
public class NearestNeighborsClient {

    private String url;
    @Setter
    @Getter
    protected String authToken;

    static {
        // Only one time
        Unirest.setObjectMapper(new ObjectMapper() {
            private org.nd4j.shade.jackson.databind.ObjectMapper jacksonObjectMapper =
                            new org.nd4j.shade.jackson.databind.ObjectMapper();

            public <T> T readValue(String value, Class<T> valueType) {
                try {
                    return jacksonObjectMapper.readValue(value, valueType);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            public String writeValue(Object value) {
                try {
                    return jacksonObjectMapper.writeValueAsString(value);
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }


    /**
     * Runs knn on the given index
     * with the given k (note that this is for data
     * already within the existing dataset not new data)
     * @param index the index of the
     *              EXISTING ndarray
     *              to run a search on
     * @param k the number of results
     * @return
     * @throws Exception
     */
    public NearestNeighborsResults knn(int index, int k) throws Exception {
        NearestNeighborRequest request = new NearestNeighborRequest();
        request.setInputIndex(index);
        request.setK(k);
        HttpRequestWithBody req = Unirest.post(url + "/knn");
        req.header("accept", "application/json")
                .header("Content-Type", "application/json").body(request);
        addAuthHeader(req);

        NearestNeighborsResults ret = req.asObject(NearestNeighborsResults.class).getBody();
        return ret;
    }

    /**
     * Run a k nearest neighbors search
     * on a NEW data point
     * @param k the number of results
     *          to retrieve
     * @param arr the array to run the search on.
     *            Note that this must be a row vector
     * @return
     * @throws Exception
     */
    public NearestNeighborsResults knnNew(int k, INDArray arr) throws Exception {
        Base64NDArrayBody base64NDArrayBody =
                        Base64NDArrayBody.builder().k(k).ndarray(Nd4jBase64.base64String(arr)).build();

        HttpRequestWithBody req = Unirest.post(url + "/knnnew");
        req.header("accept", "application/json")
                .header("Content-Type", "application/json").body(base64NDArrayBody);
        addAuthHeader(req);

        NearestNeighborsResults ret = req.asObject(NearestNeighborsResults.class).getBody();

        return ret;
    }


    /**
     * Add the specified authentication header to the specified HttpRequest
     *
     * @param request HTTP Request to add the authentication header to
     */
    protected HttpRequest addAuthHeader(HttpRequest request) {
        if (authToken != null) {
            request.header("authorization", "Bearer " + authToken);
        }

        return request;
    }

}
