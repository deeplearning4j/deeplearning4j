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

package org.deeplearning4j.eval.serde;

import org.deeplearning4j.eval.ROC;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;
import org.nd4j.shade.jackson.databind.jsontype.TypeSerializer;

import java.io.IOException;

/**
 * Custom Jackson serializer for ROC.
 * This is necessary to force calculation of the AUC and AUPRC metrics, so said metrics can be stored in the JSON;
 * this is important for exact ROC, as if it's not present at the time of serialization, it cannot be calculated later,
 * due to the underlying (very large) predictions no longer being present.
 *
 * @author Alex Black
 */
public class ROCSerializer extends JsonSerializer<ROC> {
    @Override
    public void serialize(ROC roc, JsonGenerator jsonGenerator, SerializerProvider serializerProvider)
                    throws IOException {
        if (roc.isExact()) {
            //For exact ROC implementation: force AUC and AUPRC calculation, so result can be stored in JSON, such
            //that we have them once deserialized.
            //Due to potentially huge size, exact mode doesn't store the original predictions in JSON
            roc.calculateAUC();
            roc.calculateAUCPR();
        }
        jsonGenerator.writeNumberField("thresholdSteps", roc.getThresholdSteps());
        jsonGenerator.writeNumberField("countActualPositive", roc.getCountActualPositive());
        jsonGenerator.writeNumberField("countActualNegative", roc.getCountActualNegative());
        jsonGenerator.writeObjectField("counts", roc.getCounts());
        jsonGenerator.writeNumberField("auc", roc.calculateAUC());
        jsonGenerator.writeNumberField("auprc", roc.calculateAUCPR());
        if (roc.isExact()) {
            //Store ROC and PR curves only for exact mode... they are redundant + can be calculated again for thresholded mode
            jsonGenerator.writeObjectField("rocCurve", roc.getRocCurve());
            jsonGenerator.writeObjectField("prCurve", roc.getPrecisionRecallCurve());
        }
        jsonGenerator.writeBooleanField("isExact", roc.isExact());
        jsonGenerator.writeNumberField("exampleCount", roc.getExampleCount());
        jsonGenerator.writeBooleanField("rocRemoveRedundantPts", roc.isRocRemoveRedundantPts());
    }

    @Override
    public void serializeWithType(ROC value, JsonGenerator gen, SerializerProvider serializers, TypeSerializer typeSer)
                    throws IOException {
        typeSer.writeTypePrefixForObject(value, gen);
        serialize(value, gen, serializers);
        typeSer.writeTypeSuffixForObject(value, gen);
    }
}
