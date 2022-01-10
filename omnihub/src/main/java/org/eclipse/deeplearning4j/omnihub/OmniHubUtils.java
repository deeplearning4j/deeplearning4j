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
package org.eclipse.deeplearning4j.omnihub;

import lombok.SneakyThrows;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.autodiff.samediff.SameDiff;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.*;

/**
 * Various utils for dealing with downloading files
 * and handling the omnihub cache.
 *
 * @author Adam Gibson
 */
public class OmniHubUtils {

    public final static String OMNIHUB_HOME = "OMNIHUB_HOME";
    public final static String OMNIHUB_URL = "OMNIHUB_URL";
    public static final String DEFAULT_OMNIHUB_URL = "https://raw.githubusercontent.com/KonduitAI/omnihub-zoo/main";


    /**
     * Loads a {@link MultiLayerNetwork} from the local cache
     * or downloads the file from the remote model zoo.
     * Note this is the same as calling {@link #loadNetwork(String, boolean)}
     * with a value of false
     * @param name the name of the file to load
     * @return the loaded {@link MultiLayerNetwork}
     * @throws IOException
     */
    public static MultiLayerNetwork loadNetwork(String name) throws IOException {
        return  loadNetwork(name,false);
    }


    /**
     * Loads a {@link MultiLayerNetwork} from the local cache
     * or downloads the file from the remote model zoo.
     * @param name the name of the file to load
     * @param forceDownload whether to force a new download if the file exists or not
     * @return the loaded {@link MultiLayerNetwork}
     * @throws IOException
     */
    public static MultiLayerNetwork loadNetwork(String name,boolean forceDownload) throws IOException {
        return MultiLayerNetwork.load(downloadAndLoadFromZoo("dl4j",name,forceDownload),true);
    }



    /**
     * Loads a {@link ComputationGraph} from the local cache
     * or downloads the file from the remote model zoo.
     * Note this is the same as calling {@link #loadCompGraph(String, boolean)}
     * with a value of false.
     * @param name the name of the file to load
     * @return the loaded {@link ComputationGraph}
     * @throws IOException
     */
    public static ComputationGraph loadCompGraph(String name) throws IOException {
        return loadCompGraph(name,false);
    }

    /**
     * Loads a {@link ComputationGraph} from the local cache
     * or downloads the file from the remote model zoo.
     * @param name the name of the file to load
     * @param forceDownload whether to force a new download if the file exists or not
     * @return the loaded {@link ComputationGraph}
     * @throws IOException
     */
    public static ComputationGraph loadCompGraph(String name,boolean forceDownload) throws IOException {
        return ComputationGraph.load(downloadAndLoadFromZoo("dl4j",name,forceDownload),true);
    }


    /**
     * Loads a samediff model from either the local cache
     * or downloads it to the model zoo
     * @param name the name of the file to download
     * @return
     */
    public static SameDiff loadSameDiffModel(String name) {
        return loadSameDiffModel(name,false);
    }

    /**
     * Loads a samediff model from either the local cache
     * or downloads it to the model zoo
     * @param name the name of the file to download
     * @param forceDownload whether to force the download of the file
     * @return the loaded samediff model
     */
    public static SameDiff loadSameDiffModel(String name,boolean forceDownload) {
        return SameDiff.load(downloadAndLoadFromZoo("samediff",name,forceDownload),true);
    }

    /**
     * Download and load a model from the model zoo using the given file name
     * for the given framework
     * @param framework the framework to load from
     * @param name the name of the file to load
     * @param forceDownload whether to force the download
     * @return the
     */
    @SneakyThrows
    public static File downloadAndLoadFromZoo(String framework,String name,boolean forceDownload) {
        File destination = new File(getOmnihubHome(),framework);
        File destFile = new File(destination,name);
        if(forceDownload && destFile.exists()) {
            destFile.delete();
        }
        if(!destFile.exists()) {
            String url = new StringBuilder()
                    .append(getOmnihubUrl()).append("/").append(framework).append("/").
                    append(name).toString();
            URL remoteUrl = URI.create(url).toURL();
            long size = getFileSize(remoteUrl);
            try(InputStream is = new ProgressInputStream(new BufferedInputStream(URI.create(url).toURL().openStream()),size)) {
                FileUtils.copyInputStreamToFile(is,destFile);

            } catch (MalformedURLException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return destFile;
    }


    private static int getFileSize(URL url) {
        URLConnection conn = null;
        try {
            conn = url.openConnection();
            if(conn instanceof HttpURLConnection) {
                ((HttpURLConnection)conn).setRequestMethod("HEAD");
            }
            conn.getInputStream();
            return conn.getContentLength();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            if(conn instanceof HttpURLConnection) {
                ((HttpURLConnection)conn).disconnect();
            }
        }
    }

    /**
     * Return the omnihub hurl defaulting to
     * {@link #DEFAULT_OMNIHUB_URL}
     * if the {@link #OMNIHUB_URL} is not specified.
     * @return
     */
    public static String getOmnihubUrl() {
        if(System.getenv(OMNIHUB_URL) != null) {
            return System.getenv(OMNIHUB_URL);
        } else {
            return DEFAULT_OMNIHUB_URL;
        }
    }

    /**
     * return the default omnihub home at $USER/.omnihub or
     * value of the environment variable {@link #OMNIHUB_HOME} if applicable
     * @return
     */
    public static File getOmnihubHome() {
        if(System.getenv(OMNIHUB_HOME) != null) {
            return new File(OMNIHUB_HOME);
        } else {
            return new File(System.getProperty("user.home"),".omnihub");
        }
    }
}
