/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.transform.transform;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.geo.LocationType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.geo.CoordinatesDistanceTransform;
import org.datavec.api.transform.transform.geo.IPAddressToCoordinatesTransform;
import org.datavec.api.transform.transform.geo.IPAddressToLocationTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.common.io.ClassPathResource;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author saudet
 */
public class TestGeoTransforms {

    @BeforeClass
    public static void beforeClass() throws Exception {
        //Use test resources version to avoid tests suddenly failing due to IP/Location DB content changing
        File f = new ClassPathResource("datavec-geo/GeoIP2-City-Test.mmdb").getFile();
        System.setProperty(IPAddressToLocationTransform.GEOIP_FILE_PROPERTY, f.getPath());
    }

    @AfterClass
    public static void afterClass(){
        System.setProperty(IPAddressToLocationTransform.GEOIP_FILE_PROPERTY, "");
    }

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

        String in = "81.2.69.160";
        double latitude = 51.5142;
        double longitude = -0.0931;

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
        //System.out.println(Arrays.toString(coordinates));
        assertEquals(2, coordinates.length);
        assertEquals(latitude, Double.parseDouble(coordinates[0]), 0.1);
        assertEquals(longitude, Double.parseDouble(coordinates[1]), 0.1);
    }

    @Test
    public void testIPAddressToLocationTransform() throws Exception {
        Schema schema = new Schema.Builder().addColumnString("column").build();
        LocationType[] locationTypes = LocationType.values();
        String in = "81.2.69.160";
        String[] locations = {"London", "2643743", "Europe", "6255148", "United Kingdom", "2635167",
                        "51.5142:-0.0931", "", "England", "6269131"};    //Note: no postcode in this test DB for this record

        for (int i = 0; i < locationTypes.length; i++) {
            LocationType locationType = locationTypes[i];
            String location = locations[i];

            Transform transform = new IPAddressToLocationTransform("column", locationType);
            transform.setInputSchema(schema);

            Schema out = transform.transform(schema);

            assertEquals(1, out.getColumnMetaData().size());
            assertEquals(ColumnType.String, out.getMetaData(0).getColumnType());

            List<Writable> writables = transform.map(Collections.singletonList((Writable) new Text(in)));
            assertEquals(1, writables.size());
            assertEquals(location, writables.get(0).toString());
            //System.out.println(location);
        }
    }
}
