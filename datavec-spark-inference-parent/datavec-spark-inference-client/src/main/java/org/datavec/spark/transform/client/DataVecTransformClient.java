package org.datavec.spark.transform.client;

import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import lombok.AllArgsConstructor;
import org.datavec.api.transform.TransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchRecord;
import org.datavec.spark.transform.model.CSVRecord;
import org.datavec.spark.transform.service.DataVecTransformService;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;

/**
 * Created by agibsonccc on 6/12/17.
 */
@AllArgsConstructor
public class DataVecTransformClient implements DataVecTransformService {
    private String url;

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
     * @param transformProcess
     */
    @Override
    public void setTransformProcess(TransformProcess transformProcess) {
        try {
            TransformProcess transformProcess2 = Unirest.post(url + "/transformprocess")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .body(transformProcess).asObject(TransformProcess.class).getBody();

        } catch (UnirestException e) {
            e.printStackTrace();
        }
    }

    /**
     * @return
     */
    @Override
    public TransformProcess transformProcess() {
        try {
            return  Unirest.get(url + "/transform")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .asObject(TransformProcess.class).getBody();
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param transform
     * @return
     */
    @Override
    public CSVRecord transformIncremental(CSVRecord transform) {
        try {
            CSVRecord csvRecord = Unirest.post(url + "/transformincremental")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .body(transform).asObject(CSVRecord.class).getBody();
            return csvRecord;
        } catch (UnirestException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * @param batchRecord
     * @return
     */
    @Override
    public BatchRecord transform(BatchRecord batchRecord) {
        try {
            BatchRecord batchRecord1 = Unirest.post(url + "/transform")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .body(batchRecord).asObject(BatchRecord.class).getBody();
            return batchRecord1;
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param batchRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArray(BatchRecord batchRecord) {
        try {
            Base64NDArrayBody batchArray1 = Unirest.post(url + "/transformarray")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .body(batchRecord).asObject(Base64NDArrayBody.class)
                    .getBody();
            return batchArray1;
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return null;

    }

    /**
     * @param csvRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArrayIncremental(CSVRecord csvRecord) {
        try {
            Base64NDArrayBody array = Unirest.post(url + "/transformincrementalarray")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .body(csvRecord)
                    .asObject(Base64NDArrayBody.class).getBody();
            return array;
        } catch (UnirestException e) {
            e.printStackTrace();
        }

        return null;
    }
}
