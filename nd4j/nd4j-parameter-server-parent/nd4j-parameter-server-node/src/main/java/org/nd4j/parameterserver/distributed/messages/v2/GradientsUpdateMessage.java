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

package org.nd4j.parameterserver.distributed.messages.v2;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

@NoArgsConstructor
@AllArgsConstructor
@Builder
public final class GradientsUpdateMessage implements INDArrayMessage {
    private static final long serialVersionUID = 1L;

    @Getter
    private String messageId;

    @Getter
    private INDArray payload;

    @Override
    public INDArray exec(Object context) {
        return null;
    }
}
