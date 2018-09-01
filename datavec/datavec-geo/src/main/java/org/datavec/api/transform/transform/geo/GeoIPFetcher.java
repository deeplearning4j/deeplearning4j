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

import org.apache.commons.io.FileUtils;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Downloads and caches the GeoLite2 City database created by MaxMind, available from
 * <a href="http://www.maxmind.com">http://www.maxmind.com</a> or uses one already available on system.
 *
 * @author saudet
 */
public class GeoIPFetcher {
    protected static final Logger log = LoggerFactory.getLogger(GeoIPFetcher.class);

    /** Default directory for <a href="http://dev.maxmind.com/geoip/geoipupdate/">http://dev.maxmind.com/geoip/geoipupdate/</a> */
    public static final String GEOIP_DIR = "/usr/local/share/GeoIP/";
    public static final String GEOIP_DIR2 = System.getProperty("user.home") + "/.datavec-geoip";

    public static final String CITY_DB = "GeoIP2-City.mmdb";
    public static final String CITY_LITE_DB = "GeoLite2-City.mmdb";

    public static final String CITY_LITE_URL =
                    "http://geolite.maxmind.com/download/geoip/database/GeoLite2-City.mmdb.gz";

    public static synchronized File fetchCityDB() throws IOException {
        File cityFile = new File(GEOIP_DIR, CITY_DB);
        if (cityFile.isFile()) {
            return cityFile;
        }
        cityFile = new File(GEOIP_DIR, CITY_LITE_DB);
        if (cityFile.isFile()) {
            return cityFile;
        }
        cityFile = new File(GEOIP_DIR2, CITY_LITE_DB);
        if (cityFile.isFile()) {
            return cityFile;
        }

        log.info("Downloading GeoLite2 City database...");
        File archive = new File(GEOIP_DIR2, CITY_LITE_DB + ".gz");
        File dir = new File(GEOIP_DIR2);
        dir.mkdirs();
        FileUtils.copyURLToFile(new URL(CITY_LITE_URL), archive);
        ArchiveUtils.unzipFileTo(archive.getAbsolutePath(), dir.getAbsolutePath());
        assert cityFile.isFile();

        return cityFile;
    }
}
