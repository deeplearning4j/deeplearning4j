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
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * Strategy for saving and restoring {@link NormalizerMinMaxScaler} instances in single binary files
 *
 * @author Ede Meijer
 */
public class MinMaxSerializerStrategy implements NormalizerSerializerStrategy<NormalizerMinMaxScaler> {
    @Override
    public void write(@NonNull NormalizerMinMaxScaler normalizer, @NonNull OutputStream stream) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(stream)) {
            dos.writeBoolean(normalizer.isFitLabel());
            dos.writeDouble(normalizer.getTargetMin());
            dos.writeDouble(normalizer.getTargetMax());

            Nd4j.write(normalizer.getMin(), dos);
            Nd4j.write(normalizer.getMax(), dos);

            if (normalizer.isFitLabel()) {
                Nd4j.write(normalizer.getLabelMin(), dos);
                Nd4j.write(normalizer.getLabelMax(), dos);
            }
            dos.flush();
        }
    }

    @Override
    public NormalizerMinMaxScaler restore(@NonNull InputStream stream) throws IOException {
        DataInputStream dis = new DataInputStream(stream);

        boolean fitLabels = dis.readBoolean();
        double targetMin = dis.readDouble();
        double targetMax = dis.readDouble();

        NormalizerMinMaxScaler result = new NormalizerMinMaxScaler(targetMin, targetMax);
        result.fitLabel(fitLabels);
        result.setFeatureStats(Nd4j.read(dis), Nd4j.read(dis));
        if (fitLabels) {
            result.setLabelStats(Nd4j.read(dis), Nd4j.read(dis));
        }

        return result;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.MIN_MAX;
    }
}
