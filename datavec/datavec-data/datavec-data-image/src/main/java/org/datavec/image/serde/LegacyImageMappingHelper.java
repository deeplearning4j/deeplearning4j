/*
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

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
