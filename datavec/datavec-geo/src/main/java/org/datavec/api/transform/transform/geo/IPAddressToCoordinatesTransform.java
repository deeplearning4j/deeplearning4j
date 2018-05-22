/*-
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

import org.datavec.api.transform.geo.LocationType;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.IOException;

/**
 * Uses GeoIP2 from from <a href="http://www.maxmind.com">http://www.maxmind.com</a>
 * to convert IP addresses to (approximate) coordinates (latitude:longitude).
 * For example, "128.101.101.101" becomes something like "44.9733:-93.2323".
 *
 * @author saudet
 */
public class IPAddressToCoordinatesTransform extends IPAddressToLocationTransform {

    public IPAddressToCoordinatesTransform(@JsonProperty("columnName") String columnName) throws IOException {
        this(columnName, DEFAULT_DELIMITER);
    }

    public IPAddressToCoordinatesTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("delimiter") String delimiter) throws IOException {
        super(columnName, LocationType.COORDINATES, delimiter);
    }

    @Override
    public String toString() {
        return "IPAddressToCoordinatesTransform";
    }
}
