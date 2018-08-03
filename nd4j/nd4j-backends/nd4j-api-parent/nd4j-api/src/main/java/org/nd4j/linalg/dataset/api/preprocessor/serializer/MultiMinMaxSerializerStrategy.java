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

package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Strategy for saving and restoring {@link MultiNormalizerMinMaxScaler} instances in single binary files
 *
 * @author Ede Meijer
 */
public class MultiMinMaxSerializerStrategy implements NormalizerSerializerStrategy<MultiNormalizerMinMaxScaler> {
    /**
     * Serialize a MultiNormalizerMinMaxScaler to a output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public void write(@NonNull MultiNormalizerMinMaxScaler normalizer, @NonNull OutputStream stream)
                    throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());
            dos.writeInt(normalizer.numInputs());
            dos.writeInt(normalizer.isFitLabel() ? normalizer.numOutputs() : -1);
            dos.writeDouble(normalizer.getTargetMin());
            dos.writeDouble(normalizer.getTargetMax());

            for (int i = 0; i < normalizer.numInputs(); i++) {
                Nd4j.write(normalizer.getMin(i), dos);
                Nd4j.write(normalizer.getMax(i), dos);
            }
            if (normalizer.isFitLabel()) {
                for (int i = 0; i < normalizer.numOutputs(); i++) {
                    Nd4j.write(normalizer.getLabelMin(i), dos);
                    Nd4j.write(normalizer.getLabelMax(i), dos);
                }
            }
            dos.flush();
        }
    }

    /**
     * Restore a MultiNormalizerMinMaxScaler that was previously serialized by this strategy
     *
     * @param stream the input stream to restore from
     * @return the restored MultiNormalizerMinMaxScaler
     * @throws IOException
     */
    public MultiNormalizerMinMaxScaler restore(@NonNull InputStream stream) throws IOException {
        DataInputStream dis = new DataInputStream(stream);

        boolean fitLabels = dis.readBoolean();
        int numInputs = dis.readInt();
        int numOutputs = dis.readInt();
        double targetMin = dis.readDouble();
        double targetMax = dis.readDouble();

        MultiNormalizerMinMaxScaler result = new MultiNormalizerMinMaxScaler(targetMin, targetMax);
        result.fitLabel(fitLabels);

        List<MinMaxStats> featureStats = new ArrayList<>();
        for (int i = 0; i < numInputs; i++) {
            featureStats.add(new MinMaxStats(Nd4j.read(dis), Nd4j.read(dis)));
        }
        result.setFeatureStats(featureStats);

        if (fitLabels) {
            List<MinMaxStats> labelStats = new ArrayList<>();
            for (int i = 0; i < numOutputs; i++) {
                labelStats.add(new MinMaxStats(Nd4j.read(dis), Nd4j.read(dis)));
            }
            result.setLabelStats(labelStats);
        }

        return result;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.MULTI_MIN_MAX;
    }
}
