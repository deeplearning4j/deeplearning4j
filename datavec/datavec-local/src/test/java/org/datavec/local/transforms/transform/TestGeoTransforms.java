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

package org.datavec.local.transforms.transform;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.geo.CoordinatesDistanceTransform;
import org.datavec.api.transform.transform.geo.IPAddressToCoordinatesTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author saudet
 */
public class TestGeoTransforms {

    @Test
    public void testCoordinatesDistanceTransform() throws Exception {
        Schema schema = new Schema.Builder().addColumnString("point").addColumnString("mean").addColumnString("stddev")
                        .build();

        Transform transform = new CoordinatesDistanceTransform("dist", "point", "mean", "stddev", "\\|");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);
        assertEquals(4, out.numColumns());
        assertEquals(Arrays.asList("point", "mean", "stddev", "dist"), out.getColumnNames());
        assertEquals(Arrays.asList(ColumnType.String, ColumnType.String, ColumnType.String, ColumnType.Double),
                        out.getColumnTypes());

        assertEquals(Arrays.asList((Writable) new Text("-30"), new Text("20"), new Text("10"), new DoubleWritable(5.0)),
                        transform.map(Arrays.asList((Writable) new Text("-30"), new Text("20"), new Text("10"))));
        assertEquals(Arrays.asList((Writable) new Text("50|40"), new Text("10|-20"), new Text("10|5"),
                        new DoubleWritable(Math.sqrt(160))),
                        transform.map(Arrays.asList((Writable) new Text("50|40"), new Text("10|-20"),
                                        new Text("10|5"))));
    }

    @Test
    public void testIPAddressToCoordinatesTransform() throws Exception {
        Schema schema = new Schema.Builder().addColumnString("column").build();

        Transform transform = new IPAddressToCoordinatesTransform("column", "CUSTOM_DELIMITER");
        transform.setInputSchema(schema);

        Schema out = transform.transform(schema);

        assertEquals(1, out.getColumnMetaData().size());
        assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

        String in = "128.101.101.101";
        double latitude = 44.9733;
        double longitude = -93.2323;

        List<Writable> writables = transform.map(Collections.singletonList((Writable) new Text(in)));
        assertEquals(1, writables.size());
        String[] coordinates = writables.get(0).toString().split("CUSTOM_DELIMITER");
        assertEquals(2, coordinates.length);
        assertEquals(latitude, Double.parseDouble(coordinates[0]), 0.1);
        assertEquals(longitude, Double.parseDouble(coordinates[1]), 0.1);

        //Check serialization: things like DatabaseReader etc aren't serializable, hence we need custom serialization :/
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(transform);

        byte[] bytes = baos.toByteArray();

        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        ObjectInputStream ois = new ObjectInputStream(bais);

        Transform deserialized = (Transform) ois.readObject();
        writables = deserialized.map(Collections.singletonList((Writable) new Text(in)));
        assertEquals(1, writables.size());
        coordinates = writables.get(0).toString().split("CUSTOM_DELIMITER");
        assertEquals(2, coordinates.length);
        assertEquals(latitude, Double.parseDouble(coordinates[0]), 0.1);
        assertEquals(longitude, Double.parseDouble(coordinates[1]), 0.1);
    }
}
