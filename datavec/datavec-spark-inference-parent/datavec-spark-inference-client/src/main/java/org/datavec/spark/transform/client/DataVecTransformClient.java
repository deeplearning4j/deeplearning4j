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

package org.datavec.spark.transform.client;

import com.mashape.unirest.http.ObjectMapper;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.transform.TransformProcess;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.*;
import org.datavec.spark.transform.service.DataVecTransformService;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;

/**
 * Created by agibsonccc on 6/12/17.
 */
@AllArgsConstructor
@Slf4j
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
    public void setCSVTransformProcess(TransformProcess transformProcess) {
        try {
            String s = transformProcess.toJson();
            Unirest.post(url + "/transformprocess").header("accept", "application/json")
                    .header("Content-Type", "application/json").body(s).asJson();

        } catch (UnirestException e) {
            log.error("Error in setCSVTransformProcess()", e);
            e.printStackTrace();
        }
    }

    @Override
    public void setImageTransformProcess(ImageTransformProcess imageTransformProcess) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    /**
     * @return
     */
    @Override
    public TransformProcess getCSVTransformProcess() {
        try {
            String s = Unirest.get(url + "/transformprocess").header("accept", "application/json")
                    .header("Content-Type", "application/json").asString().getBody();
            return TransformProcess.fromJson(s);
        } catch (UnirestException e) {
            log.error("Error in getCSVTransformProcess()",e);
            e.printStackTrace();
        }

        return null;
    }

    @Override
    public ImageTransformProcess getImageTransformProcess() {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    /**
     * @param transform
     * @return
     */
    @Override
    public SingleCSVRecord transformIncremental(SingleCSVRecord transform) {
        try {
            SingleCSVRecord singleCsvRecord = Unirest.post(url + "/transformincremental")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .body(transform).asObject(SingleCSVRecord.class).getBody();
            return singleCsvRecord;
        } catch (UnirestException e) {
            log.error("Error in transformIncremental(SingleCSVRecord)",e);
            e.printStackTrace();
        }
        return null;
    }


    /**
     * @param batchCSVRecord
     * @return
     */
    @Override
    public SequenceBatchCSVRecord transform(SequenceBatchCSVRecord batchCSVRecord) {
        try {
            SequenceBatchCSVRecord batchCSVRecord1 = Unirest.post(url + "/transform").header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .header(SEQUENCE_OR_NOT_HEADER,"TRUE")
                    .body(batchCSVRecord)
                    .asObject(SequenceBatchCSVRecord.class)
                    .getBody();
            return batchCSVRecord1;
        } catch (UnirestException e) {
            log.error("Error in transform(BatchCSVRecord)", e);
            e.printStackTrace();
        }

        return null;
    }
    /**
     * @param batchCSVRecord
     * @return
     */
    @Override
    public BatchCSVRecord transform(BatchCSVRecord batchCSVRecord) {
        try {
            BatchCSVRecord batchCSVRecord1 = Unirest.post(url + "/transform").header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .header(SEQUENCE_OR_NOT_HEADER,"FALSE")
                    .body(batchCSVRecord)
                    .asObject(BatchCSVRecord.class)
                    .getBody();
            return batchCSVRecord1;
        } catch (UnirestException e) {
            log.error("Error in transform(BatchCSVRecord)", e);
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param batchCSVRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArray(BatchCSVRecord batchCSVRecord) {
        try {
            Base64NDArrayBody batchArray1 = Unirest.post(url + "/transformarray").header("accept", "application/json")
                    .header("Content-Type", "application/json").body(batchCSVRecord)
                    .asObject(Base64NDArrayBody.class).getBody();
            return batchArray1;
        } catch (UnirestException e) {
            log.error("Error in transformArray(BatchCSVRecord)",e);
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param singleCsvRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformArrayIncremental(SingleCSVRecord singleCsvRecord) {
        try {
            Base64NDArrayBody array = Unirest.post(url + "/transformincrementalarray")
                    .header("accept", "application/json").header("Content-Type", "application/json")
                    .body(singleCsvRecord).asObject(Base64NDArrayBody.class).getBody();
            return array;
        } catch (UnirestException e) {
            log.error("Error in transformArrayIncremental(SingleCSVRecord)",e);
            e.printStackTrace();
        }

        return null;
    }

    @Override
    public Base64NDArrayBody transformIncrementalArray(SingleImageRecord singleImageRecord) throws IOException {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public Base64NDArrayBody transformArray(BatchImageRecord batchImageRecord) throws IOException {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    /**
     * @param singleCsvRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformSequenceArrayIncremental(BatchCSVRecord singleCsvRecord) {
        try {
            Base64NDArrayBody array = Unirest.post(url + "/transformincrementalarray")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .header(SEQUENCE_OR_NOT_HEADER,"true")
                    .body(singleCsvRecord).asObject(Base64NDArrayBody.class).getBody();
            return array;
        } catch (UnirestException e) {
            log.error("Error in transformSequenceArrayIncremental",e);
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param batchCSVRecord
     * @return
     */
    @Override
    public Base64NDArrayBody transformSequenceArray(SequenceBatchCSVRecord batchCSVRecord) {
        try {
            Base64NDArrayBody batchArray1 = Unirest.post(url + "/transformarray").header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .header(SEQUENCE_OR_NOT_HEADER,"true")
                    .body(batchCSVRecord)
                    .asObject(Base64NDArrayBody.class).getBody();
            return batchArray1;
        } catch (UnirestException e) {
            log.error("Error in transformSequenceArray",e);
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param batchCSVRecord
     * @return
     */
    @Override
    public SequenceBatchCSVRecord transformSequence(SequenceBatchCSVRecord batchCSVRecord) {
        try {
            SequenceBatchCSVRecord batchCSVRecord1 = Unirest.post(url + "/transform")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .header(SEQUENCE_OR_NOT_HEADER,"true")
                    .body(batchCSVRecord)
                    .asObject(SequenceBatchCSVRecord.class).getBody();
            return batchCSVRecord1;
        } catch (UnirestException e) {
            log.error("Error in transformSequence");
            e.printStackTrace();
        }

        return null;
    }

    /**
     * @param transform
     * @return
     */
    @Override
    public SequenceBatchCSVRecord transformSequenceIncremental(BatchCSVRecord transform) {
        try {
            SequenceBatchCSVRecord singleCsvRecord = Unirest.post(url + "/transformincremental")
                    .header("accept", "application/json")
                    .header("Content-Type", "application/json")
                    .header(SEQUENCE_OR_NOT_HEADER,"true")
                    .body(transform).asObject(SequenceBatchCSVRecord.class).getBody();
            return singleCsvRecord;
        } catch (UnirestException e) {
            log.error("Error in transformSequenceIncremental");
            e.printStackTrace();
        }
        return null;
    }
}
