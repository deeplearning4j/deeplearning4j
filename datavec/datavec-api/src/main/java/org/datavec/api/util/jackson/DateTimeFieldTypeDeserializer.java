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

package org.datavec.api.util.jackson;

import org.joda.time.DateTimeFieldType;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.DeserializationContext;
import org.nd4j.shade.jackson.databind.JsonDeserializer;
import org.nd4j.shade.jackson.databind.JsonNode;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * JsonDeserializer for deserializing Jodatime {@link DateTimeFieldType} instances
 *
 * @author Alex Black
 */
public class DateTimeFieldTypeDeserializer extends JsonDeserializer<DateTimeFieldType> {

    //Yes this is ugly - couldn't find a better way :/
    private static final Map<String, DateTimeFieldType> map = getMap();

    private static Map<String, DateTimeFieldType> getMap() {
        Map<String, DateTimeFieldType> ret = new HashMap<>();
        ret.put(DateTimeFieldType.centuryOfEra().getName(), DateTimeFieldType.centuryOfEra());
        ret.put(DateTimeFieldType.clockhourOfDay().getName(), DateTimeFieldType.clockhourOfDay());
        ret.put(DateTimeFieldType.clockhourOfHalfday().getName(), DateTimeFieldType.clockhourOfHalfday());
        ret.put(DateTimeFieldType.dayOfMonth().getName(), DateTimeFieldType.dayOfMonth());
        ret.put(DateTimeFieldType.dayOfWeek().getName(), DateTimeFieldType.dayOfWeek());
        ret.put(DateTimeFieldType.dayOfYear().getName(), DateTimeFieldType.dayOfYear());
        ret.put(DateTimeFieldType.era().getName(), DateTimeFieldType.era());
        ret.put(DateTimeFieldType.halfdayOfDay().getName(), DateTimeFieldType.halfdayOfDay());
        ret.put(DateTimeFieldType.hourOfDay().getName(), DateTimeFieldType.hourOfDay());
        ret.put(DateTimeFieldType.hourOfHalfday().getName(), DateTimeFieldType.hourOfHalfday());
        ret.put(DateTimeFieldType.millisOfDay().getName(), DateTimeFieldType.millisOfDay());
        ret.put(DateTimeFieldType.millisOfSecond().getName(), DateTimeFieldType.millisOfSecond());
        ret.put(DateTimeFieldType.minuteOfDay().getName(), DateTimeFieldType.minuteOfDay());
        ret.put(DateTimeFieldType.minuteOfHour().getName(), DateTimeFieldType.minuteOfHour());
        ret.put(DateTimeFieldType.secondOfDay().getName(), DateTimeFieldType.secondOfDay());
        ret.put(DateTimeFieldType.secondOfMinute().getName(), DateTimeFieldType.secondOfMinute());
        ret.put(DateTimeFieldType.weekOfWeekyear().getName(), DateTimeFieldType.weekOfWeekyear());
        ret.put(DateTimeFieldType.weekyear().getName(), DateTimeFieldType.weekyear());
        ret.put(DateTimeFieldType.weekyearOfCentury().getName(), DateTimeFieldType.weekyearOfCentury());
        ret.put(DateTimeFieldType.year().getName(), DateTimeFieldType.year());
        ret.put(DateTimeFieldType.yearOfCentury().getName(), DateTimeFieldType.yearOfCentury());
        ret.put(DateTimeFieldType.yearOfEra().getName(), DateTimeFieldType.yearOfEra());

        return ret;
    }

    @Override
    public DateTimeFieldType deserialize(JsonParser jsonParser, DeserializationContext deserializationContext)
                    throws IOException, JsonProcessingException {
        JsonNode node = jsonParser.getCodec().readTree(jsonParser);
        String value = node.get("fieldType").textValue();
        return map.get(value);
    }
}
