package org.deeplearning4j.eval.serde;

import com.google.common.collect.Multiset;
import org.deeplearning4j.eval.ConfusionMatrix;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.JsonSerializer;
import org.nd4j.shade.jackson.databind.SerializerProvider;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A JSON serializer for {@code ConfusionMatrix<Integer>} instances, used in {@link org.deeplearning4j.eval.Evaluation}
 *
 * @author Alex Black
 */
public class ConfusionMatrixSerializer extends JsonSerializer<ConfusionMatrix<Integer>> {
    @Override
    public void serialize(ConfusionMatrix<Integer> cm, JsonGenerator gen, SerializerProvider provider)
                    throws IOException, JsonProcessingException {
        List<Integer> classes = cm.getClasses();
        Map<Integer, Multiset<Integer>> matrix = cm.getMatrix();

        Map<Integer, int[][]> m2 = new LinkedHashMap<>();
        for (Integer i : matrix.keySet()) { //i = Actual class
            Multiset<Integer> ms = matrix.get(i);
            int[][] arr = new int[2][ms.size()];
            int used = 0;
            for (Integer j : ms.elementSet()) {
                int count = ms.count(j);
                arr[0][used] = j; //j = Predicted class
                arr[1][used] = count; //prediction count
                used++;
            }
            m2.put(i, arr);
        }

        gen.writeStartObject();
        gen.writeObjectField("classes", classes);
        gen.writeObjectField("matrix", m2);
        gen.writeEndObject();
    }
}
