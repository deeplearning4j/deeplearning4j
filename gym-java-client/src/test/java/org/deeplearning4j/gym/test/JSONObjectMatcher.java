package org.deeplearning4j.gym.test;

import org.hamcrest.Description;
import org.json.JSONObject;
import org.mockito.ArgumentMatcher;

import static org.mockito.Matchers.argThat;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/11/16.
 */


public class JSONObjectMatcher extends ArgumentMatcher<JSONObject> {
    private final JSONObject expected;

    public JSONObjectMatcher(JSONObject expected) {
        this.expected = expected;
    }

    public static JSONObject jsonEq(JSONObject expected) {
        return argThat(new JSONObjectMatcher(expected));
    }

    @Override
    public boolean matches(Object argument) {
        if (expected == null)
            return argument == null;
        if (!(argument instanceof JSONObject))
            return false;
        JSONObject actual = (JSONObject) argument;
        return expected.toString().equals(actual.toString());
    }

    @Override
    public void describeTo(Description description) {
        description.appendText(expected.toString());
    }
}
