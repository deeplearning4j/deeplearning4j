package org.deeplearning4j.gym.space;

import lombok.Value;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel on 7/8/16.
 *
 * Contain contextual information about the environment from which Observations are observed and must know how to build an Observation from json.
 *
 * @param <O> the type of Observation
 */

@Value
public class ObservationSpace<O> {

    String name;
    int[] shape;
    INDArray low;
    INDArray high;

    public ObservationSpace(JSONObject jsonObject) {

        name = jsonObject.getString("name");

        JSONArray arr = jsonObject.getJSONArray("shape");
        int lg = arr.length();

        shape = new int[lg];
        for (int i = 0; i < lg; i++) {
            this.shape[i] = arr.getInt(i);
        }

        low = Nd4j.create(shape);
        high = Nd4j.create(shape);

        JSONArray lowJson = jsonObject.getJSONArray("low");
        JSONArray highJson = jsonObject.getJSONArray("high");

        switch (shape.length){
            case 1:
                for (int i = 0; i < shape[0]; i++) {
                    low.putScalar(i, lowJson.getDouble(i));
                    high.putScalar(i, highJson.getDouble(i));
                }
                    break;
                case 2:
                    for (int i = 0; i < shape[0]; i++) {
                        JSONArray innerLowJson = lowJson.getJSONArray(i);
                        JSONArray innerHighJson = highJson.getJSONArray(i);
                        for (int j = 0; j < shape[1]; j++) {
                            //TODO when implemented in gym-http-api, check if compatible ordering
                            low.putScalar(i, j, lowJson.getDouble(i));
                            high.putScalar(i, j, highJson.getDouble(i));
                        }

                    }
                    break;
                default:
                    throw new RuntimeException("Unrecognized environment shape");
        }

    }

    public O getValue(JSONObject o, String key) {
        switch (name) {
            case "Box":
                JSONArray arr = o.getJSONArray(key);
                return (O) new Box(arr);
            default:
                throw new RuntimeException("Invalid environment name");
        }
    }

}
