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

package org.nd4j.serde.gson;

import com.google.common.primitives.Ints;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Gson  serialization
 *
 * @author Alex Black
 * @author Adam Gibson
 */
public class GsonDeserializationUtils {
    private static final JsonParser JSON_PARSER = new JsonParser();

    static {
        NumberFormat format = NumberFormat.getIntegerInstance();
        format.setGroupingUsed(false);
    }

    /**
     * Deserialize an ndarray
     * form json
     * @param serializedRawArray
     * @return
     */
    public static INDArray deserializeRawJson(String serializedRawArray) {

        //String cleanedRawArray = serializedRawArray.replaceAll("(?<=[\\d])(,)(?=[\\d])", "");
        String cleanedRawArray = serializedRawArray;
        JsonArray jsonArray = JSON_PARSER.parse(cleanedRawArray).getAsJsonArray();

        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(jsonArray.size());
        getSizeMultiDimensionalArray(jsonArray, dimensions);

        /*
            If the dimension list contains only a single element, then
            we must have an array such as [ 4, 6, 7 ] which means one row
            with columns. Since the Nd4j create method needs a minimum of two
            dimensions, then we prepend the list with 1 to designate that
            we have one row
         */
        if (isArrayWithSingleRow(dimensions)) {
            dimensions.add(0, 1);
        }

        return buildArray(dimensions, cleanedRawArray);
    }

    /*
        The below method works under the following assumption
        which is an INDArray can not have a row such as [ 1 , 2, [3, 4] ]
        and either all elements of an INDArray are either INDArrays themselves or scalars.
        So if that is the case, then it suffices to only check the first element of each JsonArray
        to see if that first element is itself an JsonArray. If it is an array, then we must check
        the first element of that array to see if it's a scalar or array.
     */

    private static void getSizeMultiDimensionalArray(JsonArray jsonArray, List<Integer> dimensions) {
        Iterator<JsonElement> iterator = jsonArray.iterator();

        if (iterator.hasNext()) {
            JsonElement jsonElement = iterator.next();
            if (jsonElement.isJsonArray()) {
                JsonArray shapeArray = jsonElement.getAsJsonArray();
                dimensions.add(shapeArray.size());
                getSizeMultiDimensionalArray(shapeArray, dimensions);
            }
        }
    }

    private static boolean isArrayWithSingleRow(List<Integer> dimensions) {
        return dimensions.size() == 1;
    }

    private static INDArray buildArray(List<Integer> dimensions, String rawArray) {
        int[] shape = Ints.toArray(dimensions);
        String[] entries = StringUtils.replacePattern(rawArray, "[\\[\\]\\n]", "").split(",");
        double[] entryValues = new double[entries.length];

        for (int i = 0; i < entries.length; i++) {
            entryValues[i] = Double.parseDouble(entries[i]);
        }

        return Nd4j.create(entryValues, shape);
    }
}
