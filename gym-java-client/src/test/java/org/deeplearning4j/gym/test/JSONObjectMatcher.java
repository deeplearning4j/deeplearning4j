package org.deeplearning4j.gym.test;

import org.json.JSONObject;
import org.mockito.ArgumentMatcher;

import static org.mockito.Matchers.argThat;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/11/16.
 */


public class JSONObjectMatcher implements ArgumentMatcher<JSONObject> {
    private final JSONObject expected;

    public JSONObjectMatcher(JSONObject expected) {
        this.expected = expected;
    }

    public static JSONObject jsonEq(JSONObject expected) {
        return argThat(new JSONObjectMatcher(expected));
    }


    @Override
    public boolean matches(JSONObject argument) {
        if (expected == null)
            return argument == null;
        return expected.toString().equals(argument.toString());    }
}
