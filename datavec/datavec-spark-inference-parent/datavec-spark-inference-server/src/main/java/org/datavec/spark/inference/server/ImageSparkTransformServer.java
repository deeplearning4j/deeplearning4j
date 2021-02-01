/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.spark.inference.server;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.ParameterException;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.TransformProcess;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.inference.model.ImageSparkTransform;
import org.datavec.spark.inference.model.model.*;
import play.BuiltInComponents;
import play.Mode;
import play.libs.Files;
import play.mvc.Http;
import play.routing.Router;
import play.routing.RoutingDsl;
import play.server.Server;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static play.mvc.Results.*;

@Slf4j
@Data
public class ImageSparkTransformServer extends SparkTransformServer {
    private ImageSparkTransform transform;

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

        if (jsonPath != null) {
            String json = FileUtils.readFileToString(new File(jsonPath));
            ImageTransformProcess transformProcess = ImageTransformProcess.fromJson(json);
            transform = new ImageSparkTransform(transformProcess);
        } else {
            log.warn("Server started with no json for transform process. Please ensure you specify a transform process via sending a post request with raw json"
                    + "to /transformprocess");
        }

        server = Server.forRouter(Mode.PROD, port, this::createRouter);
    }

    protected Router createRouter(BuiltInComponents builtInComponents){
        RoutingDsl routingDsl = RoutingDsl.fromComponents(builtInComponents);

        routingDsl.GET("/transformprocess").routingTo(req -> {
            try {
                if (transform == null)
                    return badRequest();
                log.info("Transform process initialized");
                return ok(objectMapper.writeValueAsString(transform.getImageTransformProcess())).as(contentType);
            } catch (Exception e) {
                log.error("",e);
                return internalServerError();
            }
        });

        routingDsl.POST("/transformprocess").routingTo(req -> {
            try {
                ImageTransformProcess transformProcess = ImageTransformProcess.fromJson(getJsonText(req));
                setImageTransformProcess(transformProcess);
                log.info("Transform process initialized");
                return ok(objectMapper.writeValueAsString(transformProcess)).as(contentType);
            } catch (Exception e) {
                log.error("",e);
                return internalServerError();
            }
        });

        routingDsl.POST("/transformincrementalarray").routingTo(req -> {
            try {
                SingleImageRecord record = objectMapper.readValue(getJsonText(req), SingleImageRecord.class);
                if (record == null)
                    return badRequest();
                return ok(objectMapper.writeValueAsString(transformIncrementalArray(record))).as(contentType);
            } catch (Exception e) {
                log.error("",e);
                return internalServerError();
            }
        });

        routingDsl.POST("/transformincrementalimage").routingTo(req -> {
            try {
                Http.MultipartFormData<Files.TemporaryFile> body = req.body().asMultipartFormData();
                List<Http.MultipartFormData.FilePart<Files.TemporaryFile>> files = body.getFiles();
                if (files.isEmpty() || files.get(0).getRef() == null ) {
                    return badRequest();
                }

                File file = files.get(0).getRef().path().toFile();
                SingleImageRecord record = new SingleImageRecord(file.toURI());

                return ok(objectMapper.writeValueAsString(transformIncrementalArray(record))).as(contentType);
            } catch (Exception e) {
                log.error("",e);
                return internalServerError();
            }
        });

        routingDsl.POST("/transformarray").routingTo(req -> {
            try {
                BatchImageRecord batch = objectMapper.readValue(getJsonText(req), BatchImageRecord.class);
                if (batch == null)
                    return badRequest();
                return ok(objectMapper.writeValueAsString(transformArray(batch))).as(contentType);
            } catch (Exception e) {
                log.error("",e);
                return internalServerError();
            }
        });

        routingDsl.POST("/transformimage").routingTo(req -> {
            try {
                Http.MultipartFormData<Files.TemporaryFile> body = req.body().asMultipartFormData();
                List<Http.MultipartFormData.FilePart<Files.TemporaryFile>> files = body.getFiles();
                if (files.size() == 0) {
                    return badRequest();
                }

                List<SingleImageRecord> records = new ArrayList<>();

                for (Http.MultipartFormData.FilePart<Files.TemporaryFile> filePart : files) {
                    Files.TemporaryFile file = filePart.getRef();
                    if (file != null) {
                        SingleImageRecord record = new SingleImageRecord(file.path().toUri());
                        records.add(record);
                    }
                }

                BatchImageRecord batch = new BatchImageRecord(records);

                return ok(objectMapper.writeValueAsString(transformArray(batch))).as(contentType);
            } catch (Exception e) {
                log.error("",e);
                return internalServerError();
            }
        });

        return routingDsl.build();
    }

    @Override
    public Base64NDArrayBody transformSequenceArrayIncremental(BatchCSVRecord singleCsvRecord) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Base64NDArrayBody transformSequenceArray(SequenceBatchCSVRecord batchCSVRecord) {
        throw new UnsupportedOperationException();

    }

    @Override
    public SequenceBatchCSVRecord transformSequence(SequenceBatchCSVRecord batchCSVRecord) {
        throw new UnsupportedOperationException();

    }

    @Override
    public SequenceBatchCSVRecord transformSequenceIncremental(BatchCSVRecord transform) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setCSVTransformProcess(TransformProcess transformProcess) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public void setImageTransformProcess(ImageTransformProcess imageTransformProcess) {
        this.transform = new ImageSparkTransform(imageTransformProcess);
    }

    @Override
    public TransformProcess getCSVTransformProcess() {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public ImageTransformProcess getImageTransformProcess() {
        return transform.getImageTransformProcess();
    }

    @Override
    public SingleCSVRecord transformIncremental(SingleCSVRecord singleCsvRecord) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public SequenceBatchCSVRecord transform(SequenceBatchCSVRecord batchCSVRecord) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public BatchCSVRecord transform(BatchCSVRecord batchCSVRecord) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public Base64NDArrayBody transformArray(BatchCSVRecord batchCSVRecord) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public Base64NDArrayBody transformArrayIncremental(SingleCSVRecord singleCsvRecord) {
        throw new UnsupportedOperationException("Invalid operation for " + this.getClass());
    }

    @Override
    public Base64NDArrayBody transformIncrementalArray(SingleImageRecord record) throws IOException {
        return transform.toArray(record);
    }

    @Override
    public Base64NDArrayBody transformArray(BatchImageRecord batch) throws IOException {
        return transform.toArray(batch);
    }

    public static void main(String[] args) throws Exception {
        new ImageSparkTransformServer().runMain(args);
    }
}
