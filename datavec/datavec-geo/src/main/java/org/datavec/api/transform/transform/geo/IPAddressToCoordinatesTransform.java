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
