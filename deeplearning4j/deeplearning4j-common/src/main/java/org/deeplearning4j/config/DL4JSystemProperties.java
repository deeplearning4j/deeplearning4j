package org.deeplearning4j.config;

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
     * the current working directory will be used
     * @see #CRASH_DUMP_ENABLED_PROPERTY To disable crash dump reporting
     */
    public static final String CRASH_DUMP_OUTPUT_DIRECTORY_PROPERTY = "org.deeplearning4j.crash.reporting.directory";

    /**
     * Applicability: deeplearning4j-ui<br>
     * Description: 
     */
    public static final String CHART_MAX_POINTS_PROPERTY = "org.deeplearning4j.ui.maxChartPoints";
}
