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

package org.nd4j.common.holder;

import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * A simple object mapper holder for
 * using one single {@link ObjectMapper}
 * across the whole project.
 *
 */
public class ObjectMapperHolder {

    private static ObjectMapper objectMapper = getMapper();

    private ObjectMapperHolder() {}


    /**
     * Get a single object mapper for use
     * with reading and writing json
     * @return
     */
    public static ObjectMapper getJsonMapper() {
        return objectMapper;
    }

    private static ObjectMapper getMapper() {
        ObjectMapper om = new ObjectMapper();
        //Serialize fields only, not using getters
        //Not all getters are supported - for example, UserEntity
        om.setVisibilityChecker(om.getSerializationConfig()
                .getDefaultVisibilityChecker()
                .withFieldVisibility(JsonAutoDetect.Visibility.ANY)
                .withGetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withSetterVisibility(JsonAutoDetect.Visibility.NONE)
                .withCreatorVisibility(JsonAutoDetect.Visibility.NONE));
        om.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        return om;
    }



}
