package org.nd4j.serde.gson;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class GsonDeserializationUtils {
    private static final JsonParser JSON_PARSER = new JsonParser();

    public static INDArray deserializeRawJson(String serializedRawArray) {
        JsonArray jsonArray = JSON_PARSER.parse(serializedRawArray).getAsJsonArray();

        List<Integer> dimensions = new ArrayList<>();
        dimensions.add(jsonArray.size());
        getSizeMultiDimensionalArray(jsonArray, dimensions);

        if (isArrayWithSingleRow(dimensions)) {
            dimensions.add(0, 1);
        }

        return buildArray(dimensions, serializedRawArray);
    }

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
        int rank = dimensions.size();
        int[] shape = new int[rank];
        for(int i = 0; i < rank; i++) {
            shape[i] = dimensions.get(i);
        }

        String[] entries = rawArray.replaceAll("\\[", "").replaceAll("]", "").replaceAll("\\n", "").split(",");
        double[] entryValues = new double[entries.length];

        for (int i = 0; i < entries.length; i++) {
            entryValues[i] = Double.parseDouble(entries[i]);
        }

        return Nd4j.create(entryValues, shape);
    }
}
