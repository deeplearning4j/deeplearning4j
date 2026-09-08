package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class JsonObjectPrefixSyntaxTest {
    @Test
    void rejectsMalformedPrefixesAndCompletedObjects() {
        var constraint = new JsonObjectConstraint();
        for (String text : new String[]{"{_", "{foo}", "{\"x\" 1}", "{\"x\":}",
                "{\"x\":truX}", "{\"x\":01}", "{\"x\":1,}", "{\"x\":[1,]}",
                "{\"x\":[}", "{\"x\":\"\\q\"}", "{\"x\":\"\\uQQQQ\"}",
                "{}junk", "{} {}", "{\"x\":NaN}", "{\"x\":/*c*/1}", "[]",
                "{\"x\":truX", "{\"x\":nonsense", "{\"x\":01", "{\"x\":1e+X"}) {
            assertFalse(constraint.canExtend("", text), "accepted malformed prefix: " + text);
            assertFalse(constraint.isAccepting(text), "accepted malformed object: " + text);
        }
    }

    @Test
    void permitsEveryPrefixOfValidObjectsButOnlyAcceptsCompletion() {
        var constraint = new JsonObjectConstraint();
        for (String text : new String[]{"{}", " {\"name\":\"Acme Robotics\"} ",
                "{\"café\": [true, false, null, -1.25e+2, {\"x\":\"a\\n\\u1234\"}]}",
                "{\"escaped\":\"\\\"\\\\\\/\\b\\f\\r\\t\"}"}) {
            for (int end = 0; end <= text.length(); end++) {
                assertTrue(constraint.canExtend("", text.substring(0, end)),
                        "rejected valid prefix at " + end + ": " + text);
            }
            assertTrue(constraint.isAccepting(text), text);
        }
        for (String text : new String[]{"", " ", "{", "{\"x\":", "{\"x\":tru", "{\"x\":1e+", "{\"x\":\"\\u12"}) {
            assertTrue(constraint.canExtend("", text), text);
            assertFalse(constraint.isAccepting(text), text);
        }
    }

    @Test
    void toolArgumentsRejectObservedGemmaCorruption() {
        var constraint = new ToolCallConstraint("record_organization");
        String prefix = "{\"tool\": \"record_organization\", \"args\": ";
        assertFalse(constraint.canExtend(prefix, "{_ géographique"));
        assertFalse(constraint.canExtend(prefix, "{}); later"));
        assertTrue(constraint.canExtend(prefix, "{\"name\":\"Acme Robotics\"}}"));
        assertTrue(constraint.isAccepting(prefix + "{\"name\":\"Acme Robotics\"}}"));
    }

    @Test
    void maskerNeverSelectsInvalidArgumentToken() {
        String prefix = "{\"tool\": \"record_organization\", \"args\": {";
        var masker = new ConstraintMasker(new ToolCallConstraint("record_organization"), 4);
        java.util.function.IntFunction<String> pieces = id -> new String[]{prefix, "_", "\"name\"", "}", ""}[id];
        masker.tokenEmitted(0, pieces);
        float[] masked = masker.maskLogits(new float[]{0, 100, 2, 1, 200}, 4, pieces);
        assertEquals(Float.NEGATIVE_INFINITY, masked[1]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[4]);
        assertEquals(2, masked[2]);
    }
}
