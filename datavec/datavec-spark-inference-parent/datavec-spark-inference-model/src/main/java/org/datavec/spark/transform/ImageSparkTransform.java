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

package org.datavec.spark.transform;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.transform.ImageTransformProcess;
import org.datavec.spark.transform.model.Base64NDArrayBody;
import org.datavec.spark.transform.model.BatchImageRecord;
import org.datavec.spark.transform.model.SingleImageRecord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kepricon on 17. 5. 24.
 */
@AllArgsConstructor
public class ImageSparkTransform {
    @Getter
    private ImageTransformProcess imageTransformProcess;

    public Base64NDArrayBody toArray(SingleImageRecord record) throws IOException {
        ImageWritable record2 = imageTransformProcess.transformFileUriToInput(record.getUri());
        INDArray finalRecord = imageTransformProcess.executeArray(record2);

        return new Base64NDArrayBody(Nd4jBase64.base64String(finalRecord));
    }

    public Base64NDArrayBody toArray(BatchImageRecord batch) throws IOException {
        List<INDArray> records = new ArrayList<>();

        for (SingleImageRecord imgRecord : batch.getRecords()) {
            ImageWritable record2 = imageTransformProcess.transformFileUriToInput(imgRecord.getUri());
            INDArray finalRecord = imageTransformProcess.executeArray(record2);
            records.add(finalRecord);
        }

        long shape[] = records.get(0).shape();
        INDArray array = Nd4j.create(records, new long[] {records.size(), shape[1], shape[2], shape[3]});

        return new Base64NDArrayBody(Nd4jBase64.base64String(array));
    }

}
