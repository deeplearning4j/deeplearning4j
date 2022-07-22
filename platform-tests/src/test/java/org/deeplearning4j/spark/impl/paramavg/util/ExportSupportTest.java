/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.impl.paramavg.util;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.common.resources.Downloader;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author Ede Meijer
 */
@Slf4j
public class ExportSupportTest {
    private static final String FS_CONF = "spark.hadoop.fs.defaultFS";

    @SneakyThrows
    @BeforeEach
    void before() {
        if(Platform.isWindows()) {
            File hadoopHome = new File(System.getProperty("java.io.tmpdir"),"hadoop-tmp");
            File binDir = new File(hadoopHome,"bin");
            if(!binDir.exists())
                binDir.mkdirs();
            File outputFile = new File(binDir,"winutils.exe");
            if(!outputFile.exists()) {
                log.info("Fixing spark for windows");
                Downloader.download("winutils.exe",
                        URI.create("https://github.com/cdarlint/winutils/blob/master/hadoop-2.6.5/bin/winutils.exe?raw=true").toURL(),
                        outputFile,"db24b404d2331a1bec7443336a5171f1",3);
            }

            System.setProperty("hadoop.home.dir", hadoopHome.getAbsolutePath());
        }
    }

    @Test
    public void testLocalSupported() throws IOException {
        assertSupported(new SparkConf().setMaster("local").set(FS_CONF, "file:///"));
        assertSupported(new SparkConf().setMaster("local[2]").set(FS_CONF, "file:///"));
        assertSupported(new SparkConf().setMaster("local[64]").set(FS_CONF, "file:///"));
        assertSupported(new SparkConf().setMaster("local[*]").set(FS_CONF, "file:///"));

        assertSupported(new SparkConf().setMaster("local").set(FS_CONF, "hdfs://localhost:9000"));
    }

    @Test
    public void testClusterWithRemoteFSSupported() throws IOException, URISyntaxException {
        assertSupported("spark://localhost:7077", FileSystem.get(new URI("hdfs://localhost:9000"), new Configuration()),
                        true);
    }

    @Test
    public void testClusterWithLocalFSNotSupported() throws IOException, URISyntaxException {
        assertSupported("spark://localhost:7077", FileSystem.get(new URI("file:///home/test"), new Configuration()),
                        false);
    }

    private void assertSupported(SparkConf conf) throws IOException {
        JavaSparkContext sc = new JavaSparkContext(conf.setAppName("Test").set("spark.driver.host", "localhost"));
        try {
            assertTrue(ExportSupport.exportSupported(sc));
        } finally {
            sc.stop();
        }
    }

    private void assertSupported(String master, FileSystem fs, boolean supported) throws IOException {
        assertEquals(supported, ExportSupport.exportSupported(master, fs));
    }
}
