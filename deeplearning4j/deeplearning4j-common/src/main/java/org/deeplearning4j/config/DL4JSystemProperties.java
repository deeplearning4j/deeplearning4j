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

package org.deeplearning4j.config;

/**
 * DL4JSystemProperties class contains the system properties that can be used to configure various aspects of DL4J.
 * See the javadoc of each property for details
 *
 * @author Alex Black
 */
public class DL4JSystemProperties {

    private DL4JSystemProperties(){ }

    /**
     * Applicability: Numerous modules, including deeplearning4j-datasets and deeplearning4j-zoo<br>
     * Description: Used to set the local location for downloaded remote resources such as datasets (like MNIST) and
     * pretrained models in the model zoo. Default value is set via {@code new File(System.getProperty("user.home"), ".deeplearning4j")}.
     * Setting this can be useful if the system drive has limited space/performance, a shared location for all users
     * should be used instead, or if user.home isn't set for some reason.
     */
    public static final String DL4J_RESOURCES_DIR_PROPERTY = "org.deeplearning4j.resources.directory";

    /**
     * Applicability: Numerous modules, including deeplearning4j-datasets and deeplearning4j-zoo<br>
     * Description: Used to set the base URL for hosting of resources such as datasets (like MNIST) and pretrained
     * models in the model zoo. This is provided as a fallback in case the location of these files changes; it
     * also allows for (in principle) a local mirror of these files.<br>
     * NOTE: Changing this to a location without the same files and file structure as the DL4J resource hosting is likely
     * to break external resource dowloading in DL4J!
     */
    public static final String DL4J_RESOURCES_BASE_URL_PROPERTY = "org.deeplearning4j.resources.baseurl";

    /**
     * Applicability: deeplearning4j-nn<br>
     * Description: Used for loading legacy format JSON containing custom layers. This system property is provided as an
     * alternative to {@code NeuralNetConfiguration#registerLegacyCustomClassesForJSON(Class[])}. Classes are specified in
     * comma-separated format.<br>
     * This is required ONLY when ALL of the following conditions are met:<br>
     * 1. You want to load a serialized net, saved in 1.0.0-alpha or before, AND<br>
     * 2. The serialized net has a custom Layer, GraphVertex, etc (i.e., one not defined in DL4J), AND<br>
     * 3. You haven't already called {@code NeuralNetConfiguration#registerLegacyCustomClassesForJSON(Class[])}
     */
    public static final String CUSTOM_REGISTRATION_PROPERTY = "org.deeplearning4j.config.custom.legacyclasses";

    /**
     * Applicability: deeplearning4j-nn<br>
     * Description: DL4J writes some crash dumps to disk when an OOM exception occurs - this functionality is enabled
     * by default. This is to help users identify the cause of the OOM - i.e., where native memory is actually consumed.
     * This system property can be used to disable memory crash reporting.
     * @see #CRASH_DUMP_OUTPUT_DIRECTORY_PROPERTY For configuring the output directory
     */
    public static final String CRASH_DUMP_ENABLED_PROPERTY = "org.deeplearning4j.crash.reporting.enabled";

    /**
     * Applicability: deeplearning4j-nn<br>
     * Description: DL4J writes some crash dumps to disk when an OOM exception occurs - this functionality is enabled
     * by default. This system property can be use to customize the output directory for memory crash reporting. By default,
     * the current working directory ({@code  System.getProperty("user.dir")} or {@code new File("")}) will be used
     * @see #CRASH_DUMP_ENABLED_PROPERTY To disable crash dump reporting
     */
    public static final String CRASH_DUMP_OUTPUT_DIRECTORY_PROPERTY = "org.deeplearning4j.crash.reporting.directory";

    /**
     * Applicability: deeplearning4j-ui_2.xx<br>
     * Description: The DL4J training UI (StatsListener + UIServer.getInstance().attach(ss)) will subsample the number
     * of chart points when a lot of data is present - i.e., only a maximum number of points will be shown on each chart.
     * This is to reduce the UI bandwidth requirements and client-side rendering cost.
     * To increase the number of points in charts, set this property to a larger value. Default: 512 values
     */
    public static final String CHART_MAX_POINTS_PROPERTY = "org.deeplearning4j.ui.maxChartPoints";


    /**
     * Applicability: deeplearning4j-play (deeplearning4j-ui_2.xx)<br>
     * Description: This property sets the port that the UI will be available on. Default port: 9000.
     * Set to 0 for a random port.
     */
    public static final String UI_SERVER_PORT_PROPERTY = "org.deeplearning4j.ui.port";

    /**
     * Applicability: dl4j-spark_2.xx - NTPTimeSource class (mainly used in ParameterAveragingTrainingMaster when stats
     * collection is enabled; not enabled by default)<br>
     * Description: This sets the NTP (network time protocol) server to be used when collecting stats. Default: 0.pool.ntp.org
     */
    public static final String NTP_SOURCE_SERVER_PROPERTY = "org.deeplearning4j.spark.time.NTPTimeSource.server";

    /**
     * Applicability: dl4j-spark_2.xx - NTPTimeSource class (mainly used in ParameterAveragingTrainingMaster when stats
     * collection is enabled; not enabled by default)<br>
     * Description: This sets the NTP (network time protocol) update frequency in milliseconds. Default: 1800000 (30 minutes)
     */
    public static final String NTP_SOURCE_UPDATE_FREQUENCY_MS_PROPERTY = "org.deeplearning4j.spark.time.NTPTimeSource.frequencyms";

    /**
     * Applicability: dl4j-spark_2.xx - mainly used in ParameterAveragingTrainingMaster when stats collection is enabled;
     * not enabled by default<br>
     * Description: This sets the time source to use for spark stats. Default: {@code org.deeplearning4j.spark.time.NTPTimeSource}
     */
    public static final String TIMESOURCE_CLASSNAME_PROPERTY = "org.deeplearning4j.spark.time.TimeSource";
}
