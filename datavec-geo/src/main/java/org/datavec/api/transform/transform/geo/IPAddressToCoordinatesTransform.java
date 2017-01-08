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

package org.datavec.api.transform.transform.geo;

import com.maxmind.geoip2.DatabaseReader;
import com.maxmind.geoip2.exception.GeoIp2Exception;
import com.maxmind.geoip2.model.CityResponse;
import com.maxmind.geoip2.record.Location;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetAddress;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Uses GeoIP2 from from <a href="http://www.maxmind.com">http://www.maxmind.com</a>
 * to convert IP addresses to (approximate) coordinates (latitude:longitude).
 * For example, "128.101.101.101" becomes something like "44.9733:-93.2323".
 *
 * @author saudet
 */
public class IPAddressToCoordinatesTransform extends BaseColumnTransform  {

    private static File database;
    private static DatabaseReader reader;

    public final static String DEFAULT_DELIMITER = ":";
    protected String delimiter = DEFAULT_DELIMITER;

    private static synchronized void init() throws IOException {
        // A File object pointing to your GeoIP2 or GeoLite2 database:
        // http://dev.maxmind.com/geoip/geoip2/geolite2/
        if (database == null) {
            database = GeoIPFetcher.fetchCityDB();
        }

        // This creates the DatabaseReader object, which should be reused across lookups.
        if (reader == null) {
            reader = new DatabaseReader.Builder(database).build();
        }
    }

    public IPAddressToCoordinatesTransform(@JsonProperty("columnName") String columnName) throws IOException {
        this(columnName, DEFAULT_DELIMITER);
    }

    public IPAddressToCoordinatesTransform(@JsonProperty("columnName") String columnName,
            @JsonProperty("delimiter") String delimiter) throws IOException {
        super(columnName);
        this.delimiter = delimiter;
        init();
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new StringMetaData(newName); //Output after transform: String (Text)
    }

    @Override
    public Writable map(Writable columnWritable) {
        try {
            InetAddress ipAddress = InetAddress.getByName(columnWritable.toString());
            CityResponse response = reader.city(ipAddress);
            Location location = response.getLocation();
            return new Text(location.getLatitude() + delimiter + location.getLongitude());
        } catch (GeoIp2Exception | IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return "IPAddressToCoordinatesTransform";
    }

    //Custom serialization methods, because GeoIP2 doesn't allow DatabaseReader objects to be serialized :(
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        init();
    }

    @Override
    public Object map(Object input) {
        return null;
    }
}
