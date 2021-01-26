/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.remote.helpers;

import com.google.gson.Gson;
import lombok.*;
import org.nd4j.remote.clients.serde.JsonDeserializer;
import org.nd4j.remote.clients.serde.JsonSerializer;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class House {
    private int district;
    private int bedrooms;
    private int bathrooms;
    private int area;


    public static class HouseSerializer implements JsonSerializer<House> {
        @Override
        public String serialize(@NonNull House o) {
            return new Gson().toJson(o);
        }
    }

    public static class HouseDeserializer implements JsonDeserializer<House> {
        @Override
        public House deserialize(@NonNull String json) {
            return new Gson().fromJson(json, House.class);
        }
    }
}
