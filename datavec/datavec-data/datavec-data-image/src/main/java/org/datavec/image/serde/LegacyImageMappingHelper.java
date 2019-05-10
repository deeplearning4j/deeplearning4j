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

package org.datavec.image.serde;

import org.datavec.api.transform.serde.legacy.GenericLegacyDeserializer;
import org.datavec.api.transform.serde.legacy.LegacyMappingHelper;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

public class LegacyImageMappingHelper {

    @JsonDeserialize(using = LegacyImageTransformDeserializer.class)
    public static class ImageTransformHelper { }

    public static class LegacyImageTransformDeserializer extends GenericLegacyDeserializer<ImageTransform> {
        public LegacyImageTransformDeserializer() {
            super(ImageTransform.class, LegacyMappingHelper.getLegacyMappingImageTransform());
        }
    }
    
}
