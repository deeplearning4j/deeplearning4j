package org.nd4j.remote;

import lombok.val;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.HttpClientBuilder;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;
import org.nd4j.adapters.InferenceAdapter;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class SameDiffServletTest {

    private SameDiffJsonModelServer server;

    @Before
    public void setUp() throws Exception {
        server = new SameDiffJsonModelServer.Builder<String, String>()
                .sdModel(SameDiff.create())
                .port(8080)
                .inferenceAdapter(new InferenceAdapter<String, String>() {
                    @Override
                    public MultiDataSet apply(String input) {
                        return null;
                    }

                    @Override
                    public String apply(INDArray... nnOutput) {
                        return null;
                    }
                })
                .outputSerializer(new JsonSerializer<String>() {
                    @Override
                    public String serialize(String o) {
                        return "";
                    }
                })
                .inputDeserializer(new JsonDeserializer<String>() {
                    @Override
                    public String deserialize(String json) {
                        return "";
                    }
                })
                .orderedOutputNodes(new String[]{"output"})
                .build();

        server.start();
        //server.join();
    }

    @After
    public void tearDown() throws Exception {
        server.stop();
    }

    @Test
    public void getEndpoints() throws IOException {
        val request = new HttpGet( "http://localhost:8080/v1" );
        request.setHeader("Content-type", "application/json");

        val response = HttpClientBuilder.create().build().execute( request );
        assertEquals(200, response.getStatusLine().getStatusCode());
    }

    @Test
    public void testContentTypeGet() throws IOException {
        val request = new HttpGet( "http://localhost:8080/v1" );
        request.setHeader("Content-type", "text/plain");

        val response = HttpClientBuilder.create().build().execute( request );
        assertEquals(415, response.getStatusLine().getStatusCode());
    }

    @Test
    public void testContentTypePost() throws Exception {
        val request = new HttpPost("http://localhost:8080/v1/serving");
        request.setHeader("Content-type", "text/plain");
        val response = HttpClientBuilder.create().build().execute( request );
        assertEquals(415, response.getStatusLine().getStatusCode());
    }

    @Test
    public void postForServing() throws Exception {
        val request = new HttpPost("http://localhost:8080/v1/serving");
        request.setHeader("Content-type", "application/json");
        val response = HttpClientBuilder.create().build().execute( request );
        assertEquals(500, response.getStatusLine().getStatusCode());
    }

    @Test
    public void testNotFoundPost() throws Exception {
        val request = new HttpPost("http://localhost:8080/v1/serving/some");
        request.setHeader("Content-type", "application/json");
        val response = HttpClientBuilder.create().build().execute( request );
        assertEquals(404, response.getStatusLine().getStatusCode());
    }

    @Test
    public void testNotFoundGet() throws Exception {
        val requestGet = new HttpGet( "http://localhost:8080/v1/not_found" );
        requestGet.setHeader("Content-type", "application/json");

        val responseGet = HttpClientBuilder.create().build().execute( requestGet );
        assertEquals(404, responseGet.getStatusLine().getStatusCode());
    }

}
