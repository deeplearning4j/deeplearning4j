package org.datavec.python;

import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.JsonSerializer;
import org.datavec.api.transform.serde.YamlSerializer;
import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class TestSerde {

    public static YamlSerializer y = new YamlSerializer();
    public static JsonSerializer j = new JsonSerializer();

    @Test
    public void testBasicSerde() throws Exception{
        Schema schema = new Schema.Builder()
                .addColumnInteger("col1")
                .addColumnFloat("col2")
                .addColumnString("col3")
                .addColumnDouble("col4")
                .build();

        Transform t = new PythonTransform(
                "col1+=3\ncol2+=2\ncol3+='a'\ncol4+=2.0",
                schema
        );

        String yaml = y.serialize(t);
        String json = j.serialize(t);

        Transform t2 = y.deserializeTransform(json);
        Transform t3 = j.deserializeTransform(json);
        assertEquals(t, t2);
        assertEquals(t, t3);
    }

}
