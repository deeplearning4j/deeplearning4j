package org.nd4j.versioncheck;

import lombok.extern.slf4j.Slf4j;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;
import org.reflections.scanners.SubTypesScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;

import java.util.*;
import java.util.regex.Pattern;

/**
 * A runtime version check utility that does 2 things:<br>
 * (a) validates the versions of ND4J, DL4J, DataVec, RL4J, Arbiter on the class path, logging a warning if
 * incompatible versions are found<br>
 * (b) allows users to get version information for the above projects at runtime.
 *
 * @author Alex Black
 */
@Slf4j
public class VersionCheck {

    /**
     * Setting the system property to false will stop ND4J from performing the version check, and logging any
     * warnings/errors. By default, the version check is unable.
     */
    public static final String VERSION_CHECK_PROPERTY = "org.nd4j.versioncheck";

    private static final String SCALA_210_SUFFIX = "_2.10";
    private static final String SCALA_211_SUFFIX = "_2.11";
    private static final String SPARK_1_VER_STRING = "spark_1";
    private static final String SPARK_2_VER_STRING = "spark_2";

    private static final String UNKNOWN_VERSION = "(Unknown, pre-0.9.1)";
    private static final String UNKNOWN_VERSION_2 = "(Unknown)";

    private static final String DL4J_GROUPID = "org.deeplearning4j";
    private static final String DL4J_ARTIFACT = "deeplearning4j-nn";
    private static final String DL4J_CLASS = "org.deeplearning4j.nn.conf.MultiLayerConfiguration";

    private static final String DATAVEC_GROUPID = "org.datavec";
    private static final String DATAVEC_ARTIFACT = "datavec-api";
    private static final String DATAVEC_CLASS = "org.datavec.api.writable.DoubleWritable";

    private static final String ND4J_GROUPID = "org.nd4j";

    private static final String ND4J_JBLAS_CLASS = "org.nd4j.linalg.jblas.JblasBackend";
    private static final String CANOVA_CLASS = "org.canova.api.io.data.DoubleWritable";

    private static final Set<String> GROUPIDS_TO_CHECK = new HashSet<>(Arrays.asList(
            ND4J_GROUPID, DL4J_GROUPID, DATAVEC_GROUPID));    //NOTE: DL4J_GROUPID also covers Arbiter and RL4J

    /**
     * Detailed level for logging:
     * GAV: display group ID, artifact, version
     * GAVC: display group ID, artifact, version, commit ID
     * FULL: display group ID, artifact, version, commit ID, build time, branch, commit message
     */
    public enum Detail {
        GAV,
        GAVC,
        FULL
    }

    private VersionCheck(){

    }

    /**
     * Perform a check of the versions of ND4J, DL4J, DataVec, RL4J and Arbiter dependencies, logging a warning
     * if necessary.
     */
    public static void checkVersions(){
        boolean doCheck = Boolean.parseBoolean(System.getProperty(VERSION_CHECK_PROPERTY, "true"));

        if(!doCheck){
            return;
        }

        if(classExists(ND4J_JBLAS_CLASS)) {
            //nd4j-jblas is ancient and incompatible
            log.error("Found incompatible/obsolete backend and version (nd4j-jblas) on classpath. ND4J is unlikely to"
                    + " function correctly with nd4j-jblas on the classpath. JVM will now exit.");
            System.exit(1);
        }

        if(classExists(CANOVA_CLASS)) {
            //Canova is ancient and likely to pull in incompatible dependencies
            log.error("Found incompatible/obsolete library Canova on classpath. ND4J is unlikely to"
                    + " function correctly with this library on the classpath. JVM will now exit.");
            System.exit(1);
        }

        List<VersionInfo> dependencies = getVersionInfos();
        if(dependencies.size() <= 2){
            //No -properties.git files were found on the classpath. This may be due to a misconfigured uber-jar
            // or maybe running in IntelliJ with "dynamic.classpath" set to true (in workspace.xml). Either way,
            // we can't check versions and don't want to log an error, which will more often than not be wrong
            if(dependencies.size() == 0){
                return;
            }

            //Another edge case: no -properties.git files were found, but DL4J and/or DataVec were inferred
            // by class names. If these "inferred by opName" versions were the only things found, we should also
            // not log a warning, as we can't check versions in this case

            boolean dl4jViaClass = false;
            boolean datavecViaClass = false;
            for(VersionInfo vi : dependencies ){
                if(DL4J_GROUPID.equals(vi.getGroupId()) && DL4J_ARTIFACT.equals(vi.getArtifactId())
                        && (UNKNOWN_VERSION.equals(vi.getBuildVersion()) || UNKNOWN_VERSION_2.equals(vi.getBuildVersion()))){
                    dl4jViaClass = true;
                } else if(DATAVEC_GROUPID.equals(vi.getGroupId()) && DATAVEC_ARTIFACT.equals(vi.getArtifactId())
                        && (UNKNOWN_VERSION.equals(vi.getBuildVersion()) || UNKNOWN_VERSION_2.equals(vi.getBuildVersion()))){
                    datavecViaClass = true;
                }
            }

            if(dependencies.size() == 1 && (dl4jViaClass || datavecViaClass)){
                return;
            } else if(dependencies.size() == 2 && dl4jViaClass && datavecViaClass){
                return;
            }
        }

        Set<String> foundVersions = new HashSet<>();
        for(VersionInfo vi : dependencies){
            String g = vi.getGroupId();
            if(g != null && GROUPIDS_TO_CHECK.contains(g)){
                String version = vi.getBuildVersion();

                if(version.contains("_spark_")){
                    //Normalize spark versions:
                    // "0.9.1_spark_1" to "0.9.1" and "0.9.1_spark_1-SNAPSHOT" to "0.9.1-SNAPSHOT"
                    version = version.replaceAll("_spark_1","");
                    version = version.replaceAll("_spark_2","");
                }

                foundVersions.add(version);
            }
        }

        boolean logVersions = false;

        if(foundVersions.size() > 1){
            log.warn("*** ND4J VERSION CHECK FAILED - INCOMPATIBLE VERSIONS FOUND ***");
            log.warn("Incompatible versions (different version number) of DL4J, ND4J, RL4J, DataVec, Arbiter are unlikely to function correctly");
            logVersions = true;
        } else if(foundVersions.size() == 1 && foundVersions.contains(UNKNOWN_VERSION)){
            log.warn("*** ND4J VERSION CHECK FAILED - COULD NOT INFER VERSIONS ***");
            log.warn("Incompatible versions (different version number) of DL4J, ND4J, RL4J, DataVec, Arbiter are unlikely to function correctly");
            logVersions = true;
        }


        //Also: check for mixed scala versions - but only for our dependencies... These are in the artifact ID,
        // scored like dl4j-spack_2.10 and deeplearning4j-ui_2.11
        //And check for mixed spark versions (again, just DL4J/DataVec etc dependencies for now)
        boolean scala210 = false;
        boolean scala211 = false;
        boolean spark1 = false;
        boolean spark2 = false;
        for(VersionInfo vi : dependencies){
            String artifact = vi.getArtifactId();
            if(!scala210 && artifact.contains(SCALA_210_SUFFIX)){
                scala210 = true;
            }
            if(!scala211 && artifact.contains(SCALA_211_SUFFIX)){
                scala211 = true;
            }

            String version = vi.getBuildVersion();
            if(!spark1 && version.contains(SPARK_1_VER_STRING)){
                spark1 = true;
            }
            if(!spark2 && version.contains(SPARK_2_VER_STRING)){
                spark2 = true;
            }
        }

        if(scala210 && scala211){
            log.warn("*** ND4J VERSION CHECK FAILED - FOUND BOTH SCALA VERSION 2.10 AND 2.11 ARTIFACTS ***");
            log.warn("Projects with mixed Scala versions (2.10/2.11) are unlikely to function correctly");
            logVersions = true;
        }

        if(spark1 && spark2){
            log.warn("*** ND4J VERSION CHECK FAILED - FOUND BOTH SPARK VERSION 1 AND 2 ARTIFACTS ***");
            log.warn("Projects with mixed Spark versions (1 and 2) are unlikely to function correctly");
            logVersions = true;
        }

        if(logVersions){
            log.info("Versions of artifacts found on classpath:");
            logVersionInfo();
        }
    }

    /**
     * @return A list of the property files containing the build/version info
     */
    public static List<String> listGitPropertiesFiles() {
        Reflections reflections = new Reflections(new ConfigurationBuilder().filterInputsBy(
                new FilterBuilder().exclude(".*").include("/ai/skymind/*")).setScanners(new ResourcesScanner()));
        Set<String> resources = reflections.getResources(Pattern.compile(".*-git.properties"));

        List<String> out = new ArrayList<>(resources);
        Collections.sort(out);      //Equivalent to sorting by groupID and artifactID

        return out;
    }

    /**
     * @return A list containing the information for the discovered dependencies
     */
    public static List<VersionInfo> getVersionInfos() {

        boolean dl4jFound = false;
        boolean datavecFound = false;

        List<VersionInfo> repState = new ArrayList<>();
        for(String s : listGitPropertiesFiles()){
            VersionInfo grs;

            try{
                grs = new VersionInfo(s);
            } catch (Exception e){
                log.warn("Error reading property files for {}", s);
                continue;
            }
            repState.add(grs);

            if(!dl4jFound && DL4J_GROUPID.equalsIgnoreCase(grs.getGroupId()) && DL4J_ARTIFACT.equalsIgnoreCase(grs.getArtifactId())){
                dl4jFound = true;
            }

            if(!datavecFound && DATAVEC_GROUPID.equalsIgnoreCase(grs.getGroupId()) && DATAVEC_ARTIFACT.equalsIgnoreCase(grs.getArtifactId())){
                datavecFound = true;
            }
        }

        //Note that if NO git.properties files were found, it's still possible that the DL4J/DataVec versions found
        // by their class names are correct. Consequently, only call them "pre-0.9.1" if we can be sure that's the case,
        // otherwise just call them "Unknown"
        String unknownVersionString = repState.size() == 0 ? UNKNOWN_VERSION_2 : UNKNOWN_VERSION;

        if(!dl4jFound){
            //See if pre-0.9.1 DL4J is present on classpath;
            if(classExists(DL4J_CLASS)){
                List<VersionInfo> temp = new ArrayList<>();
                temp.add(new VersionInfo(DL4J_GROUPID, DL4J_ARTIFACT, unknownVersionString));
                temp.addAll(repState);
                repState = temp;
            }
        }

        if(!datavecFound){
            //See if pre-0.9.1 DataVec is present on classpath
            if(classExists(DATAVEC_CLASS)){
                List<VersionInfo> temp = new ArrayList<>();
                temp.add(new VersionInfo(DATAVEC_GROUPID, DATAVEC_ARTIFACT, unknownVersionString));
                temp.addAll(repState);
                repState = temp;
            }
        }

        if(classExists(ND4J_JBLAS_CLASS)){
            //nd4j-jblas is ancient and incompatible
            log.error("Found incompatible/obsolete backend and version (nd4j-jblas) on classpath. ND4J is unlikely to"
                    + " function correctly with nd4j-jblas on the classpath.");
        }

        if(classExists(CANOVA_CLASS)){
            //Canova is anchient and likely to pull in incompatible
            log.error("Found incompatible/obsolete library Canova on classpath. ND4J is unlikely to"
                    + " function correctly with this library on the classpath.");
        }


        return repState;
    }

    private static boolean classExists(String className){
        try{
            Class.forName(className);
            return true;
        } catch (ClassNotFoundException e ){
            //OK - not found
        }
        return false;
    }

    /**
     * @return A string representation of the version information, with the default (GAV) detail level
     */
    public static String versionInfoString() {
        return versionInfoString(Detail.GAV);
    }

    /**
     * Get the version information for dependencies as a string with a specified amount of detail
     *
     * @param detail Detail level for the version information. See {@link Detail}
     * @return Version information, as a String
     */
    public static String versionInfoString(Detail detail) {
        StringBuilder sb = new StringBuilder();
        for(VersionInfo grp : getVersionInfos()){
            sb.append(grp.getGroupId()).append(" : ").append(grp.getArtifactId()).append(" : ").append(grp.getBuildVersion());
            switch (detail){
                case FULL:
                case GAVC:
                    sb.append(" - ").append(grp.getCommitIdAbbrev());
                    if(detail != Detail.FULL) break;

                    sb.append("buildTime=").append(grp.getBuildTime()).append("branch=").append(grp.getBranch())
                            .append("commitMsg=").append(grp.getCommitMessageShort());
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Log of the version information with the default level of detail
     */
    public static void logVersionInfo(){
        logVersionInfo(Detail.GAV);
    }

    /**
     * Log the version information with the specified level of detail
     * @param detail Level of detail for logging
     */
    public static void logVersionInfo(Detail detail){

        List<VersionInfo> info = getVersionInfos();

        for(VersionInfo grp : info){
            switch (detail){
                case GAV:
                    log.info("{} : {} : {}", grp.getGroupId(), grp.getArtifactId(), grp.getBuildVersion());
                    break;
                case GAVC:
                    log.info("{} : {} : {} - {}", grp.getGroupId(), grp.getArtifactId(), grp.getBuildVersion(),
                            grp.getCommitIdAbbrev());
                    break;
                case FULL:
                    log.info("{} : {} : {} - {}, buildTime={}, buildHost={} branch={}, commitMsg={}", grp.getGroupId(), grp.getArtifactId(),
                            grp.getBuildVersion(), grp.getCommitId(), grp.getBuildTime(), grp.getBuildHost(),
                            grp.getBranch(), grp.getCommitMessageShort());
                    break;
            }
        }
    }
}
