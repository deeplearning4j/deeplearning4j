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

package org.datavec.api.transform.transform.geo;

import com.maxmind.geoip2.DatabaseReader;
import com.maxmind.geoip2.exception.GeoIp2Exception;
import com.maxmind.geoip2.model.CityResponse;
import com.maxmind.geoip2.record.Location;
import com.maxmind.geoip2.record.Subdivision;
import org.datavec.api.transform.geo.LocationType;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetAddress;

/**
 * Uses GeoIP2 from from <a href="http://www.maxmind.com">http://www.maxmind.com</a>
 * to convert IP addresses to (approximate) locations.
 *
 * @see LocationType
 *
 * @author saudet
 */
public class IPAddressToLocationTransform extends BaseColumnTransform {

    private static File database;
    private static DatabaseReader reader;

    public final static String DEFAULT_DELIMITER = ":";
    protected String delimiter = DEFAULT_DELIMITER;
    protected LocationType locationType;

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

    public IPAddressToLocationTransform(String columnName) throws IOException {
        this(columnName, LocationType.CITY);
    }

    public IPAddressToLocationTransform(String columnName, LocationType locationType) throws IOException {
        this(columnName, locationType, DEFAULT_DELIMITER);
    }

    public IPAddressToLocationTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("delimiter") LocationType locationType, @JsonProperty("delimiter") String delimiter)
                    throws IOException {
        super(columnName);
        this.delimiter = delimiter;
        this.locationType = locationType;
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
            String text = "";
            switch (locationType) {
                case CITY:
                    text = response.getCity().getName();
                    break;
                case CITY_ID:
                    text = response.getCity().getGeoNameId().toString();
                    break;
                case CONTINENT:
                    text = response.getContinent().getName();
                    break;
                case CONTINENT_ID:
                    text = response.getContinent().getGeoNameId().toString();
                    break;
                case COUNTRY:
                    text = response.getCountry().getName();
                    break;
                case COUNTRY_ID:
                    text = response.getCountry().getGeoNameId().toString();
                    break;
                case COORDINATES:
                    Location location = response.getLocation();
                    text = location.getLatitude() + delimiter + location.getLongitude();
                    break;
                case POSTAL_CODE:
                    text = response.getPostal().getCode();
                    break;
                case SUBDIVISIONS:
                    for (Subdivision s : response.getSubdivisions()) {
                        if (text.length() > 0) {
                            text += delimiter;
                        }
                        text += s.getName();
                    }
                    break;
                case SUBDIVISIONS_ID:
                    for (Subdivision s : response.getSubdivisions()) {
                        if (text.length() > 0) {
                            text += delimiter;
                        }
                        text += s.getGeoNameId().toString();
                    }
                    break;
                default:
                    assert false;
            }
            return new Text(text);
        } catch (GeoIp2Exception | IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return "IPAddressToLocationTransform";
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
