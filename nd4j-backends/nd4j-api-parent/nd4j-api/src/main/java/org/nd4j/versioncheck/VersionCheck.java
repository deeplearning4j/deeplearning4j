package org.nd4j.versioncheck;

import lombok.extern.slf4j.Slf4j;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;

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

    private static final String UNKNOWN_VERSION = "(Unknown, pre-0.9.1)";

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
     * GAV: display group ID, artefact, version
     * GAVC: display group ID, artefact, version, commit ID
     * FULL: display group ID, artefact, version, commit ID, build time, branch, commit message
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

        if(classExists(ND4J_JBLAS_CLASS)){
            //nd4j-jblas is ancient and incompatible
            log.error("Found incompatible/obsolete backend and version (nd4j-jblas) on classpath. ND4J is unlikely to"
                    + " function correctly with nd4j-jblas on the classpath. JVM will now exit.");
            System.exit(1);
        }

        if(classExists(CANOVA_CLASS)){
            //Canova is ancient and likely to pull in incompatible dependencies
            log.error("Found incompatible/obsolete library Canova on classpath. ND4J is unlikely to"
                    + " function correctly with this library on the classpath. JVM will now exit.");
            System.exit(1);
        }

        List<VersionInfo> repos = getVersionInfos();
        Set<String> foundVersions = new HashSet<>();
        for(VersionInfo gpr : repos){
            String g = gpr.getGroupId();
            if(g != null && GROUPIDS_TO_CHECK.contains(g)){
                foundVersions.add(gpr.getBuildVersion());
            }
        }

        if(foundVersions.size() > 1){
            log.warn("*** ND4J VERSION CHECK FAILED - INCOMPATIBLE VERSIONS FOUND ***");
            log.warn("Incompatible versions (different version number) of DL4J, ND4J, RL4J, DataVec, Arbiter are unlikely to function correctly");
            log.info("Versions of artifacts found on classpath:");
            logVersionInfo();
        } else if(foundVersions.size() == 1 && foundVersions.contains(UNKNOWN_VERSION)){
            log.warn("*** ND4J VERSION CHECK FAILED - COULD NOT INFER VERSIONS ***");
            log.warn("Incompatible versions (different version number) of DL4J, ND4J, RL4J, DataVec, Arbiter are unlikely to function correctly");
            log.info("Versions of artifacts found on classpath:");
            logVersionInfo();
        }
    }

    /**
     * @return A list of the property files containing the build/version info
     */
    public static List<String> listGitPropertiesFiles() {
        Reflections reflections = new Reflections(new ResourcesScanner());

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

        if(!dl4jFound){
            //See if pre-0.9.1 DL4J is present on classpath;
            if(classExists(DL4J_CLASS)){
                repState.add(new VersionInfo(DL4J_GROUPID, DL4J_ARTIFACT, UNKNOWN_VERSION));
            }
        }

        if(!datavecFound){
            //See if pre-0.9.1 DataVec is present on classpath
            if(classExists(DATAVEC_CLASS)){
                repState.add(new VersionInfo(DATAVEC_GROUPID, DATAVEC_ARTIFACT, UNKNOWN_VERSION));
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
     * @return
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
