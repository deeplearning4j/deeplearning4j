/*
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.reduce.impl;

import lombok.Data;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.ops.IAggregableReduceOp;
import org.datavec.api.transform.reduce.AggregableColumnReduction;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.base.Preconditions;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.List;

/**
 * Given a set of latitude/longitude coordinates, encoded in {@link Text} writables with format "lat,long" (the
 * delimiter is configurable), determine the geographic midpoint.
 * See "geographic midpoint" at: http://www.geomidpoint.com/methods.html
 * For implementation algorithm, see: http://www.geomidpoint.com/calculation.html
 *
 * @author Alex Black
 */
@Data
public class GeographicMidpointReduction implements AggregableColumnReduction {

    public static final double EDGE_CASE_EPS = 1e-9;

    private String delim;
    private String newColumnName;

    /**
     * @param delim Delimiter for the coordinates in text format. For example, if format is "lat,long" use ","
     */
    public GeographicMidpointReduction(String delim) {
        this(delim, null);
    }

    public GeographicMidpointReduction(@JsonProperty("delim") String delim, @JsonProperty("newColumnName") String newColumnName){
        this.delim = delim;
        this.newColumnName = newColumnName;
    }

    @Override
    public IAggregableReduceOp<Writable, List<Writable>> reduceOp() {
        return new AverageCoordinateReduceOp(delim);
    }

    @Override
    public List<String> getColumnsOutputName(String columnInputName) {
        if(newColumnName != null){
            return Collections.singletonList(newColumnName);
        }
        return Collections.singletonList("midpoint(" + columnInputName + ")");
    }

    @Override
    public List<ColumnMetaData> getColumnOutputMetaData(List<String> newColumnName, ColumnMetaData columnInputMeta) {
        return Collections.<ColumnMetaData>singletonList(new StringMetaData(newColumnName.get(0)));
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //No change
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        //No op
    }

    @Override
    public Schema getInputSchema() {
        return null;
    }

    @Override
    public String outputColumnName() {
        return null;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[0];
    }

    @Override
    public String[] columnNames() {
        return new String[0];
    }

    @Override
    public String columnName() {
        return null;
    }

    public static class AverageCoordinateReduceOp implements IAggregableReduceOp<Writable, List<Writable>> {
        private static final double PI_180 = Math.PI / 180.0;

        private String delim;

        private double sumx;
        private double sumy;
        private double sumz;
        private int count;

        public AverageCoordinateReduceOp(String delim){
            this.delim = delim;
        }

        @Override
        public <W extends IAggregableReduceOp<Writable, List<Writable>>> void combine(W accu) {
            if(accu instanceof AverageCoordinateReduceOp){
                AverageCoordinateReduceOp r = (AverageCoordinateReduceOp)accu;
                sumx += r.sumx;
                sumy += r.sumy;
                sumz += r.sumz;
                count += r.count;
            } else {
                throw new IllegalStateException("Cannot combine type of class: " + accu.getClass());
            }
        }

        @Override
        public void accept(Writable writable) {
            String str = writable.toString();
            String[] split = str.split(delim);
            if(split.length != 2){
                throw new IllegalStateException("Could not parse lat/long string: \"" + str + "\"" );
            }
            double latDeg = Double.parseDouble(split[0]);
            double longDeg = Double.parseDouble(split[1]);

            Preconditions.checkState(latDeg >= -90.0 && latDeg <= 90.0, "Invalid latitude: must be -90 to -90. Got: %s", latDeg);
            Preconditions.checkState(latDeg >= -180.0 && latDeg <= 180.0, "Invalid longitude: must be -180 to -180. Got: %s", longDeg);

            double lat = latDeg * PI_180;
            double lng = longDeg * PI_180;

            double x = Math.cos(lat) * Math.cos(lng);
            double y = Math.cos(lat) * Math.sin(lng);
            double z = Math.sin(lat);

            sumx += x;
            sumy += y;
            sumz += z;
            count++;
        }

        @Override
        public List<Writable> get() {
            double x = sumx / count;
            double y = sumy / count;
            double z = sumz / count;

            if(count == 0){
                throw new IllegalStateException("Cannot calculate geographic midpoint: no datapoints were added to be reduced");
            }

            if(Math.abs(x) < EDGE_CASE_EPS && Math.abs(y) < EDGE_CASE_EPS && Math.abs(z) < EDGE_CASE_EPS ){
                throw new IllegalStateException("No Geographic midpoint exists: midpoint is center of the earth");
            }

            double longRad = Math.atan2(y,x);
            double hyp = Math.sqrt(x*x + y*y);
            double latRad = Math.atan2(z, hyp);

            double latDeg = latRad / PI_180;
            double longDeg = longRad / PI_180;

            Preconditions.checkState(!Double.isNaN(latDeg), "Final latitude is NaN");
            Preconditions.checkState(!Double.isNaN(longDeg), "Final longitude is NaN");

            String str = latDeg + delim + longDeg;
            return Collections.<Writable>singletonList(new Text(str));
        }
    }
}
