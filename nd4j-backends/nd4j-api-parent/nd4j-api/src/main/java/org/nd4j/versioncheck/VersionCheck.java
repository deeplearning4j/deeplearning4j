package org.nd4j.versioncheck;

import lombok.extern.slf4j.Slf4j;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;

import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by Alex on 04/08/2017.
 */
@Slf4j
public class VersionCheck {

    public static final String VERSION_CHECK_PROPERTY = "org.nd4j.versioncheck";

    private static final String UNKNOWN_VERSION = "(Unknown, pre-0.9.1)";

    private static final String DL4J_GROUPID = "org.deeplearning4j";
    private static final String DL4J_ARTIFACT = "deeplearning4j-nn";
    private static final String DL4J_CLASS = "org.deeplearning4j.nn.conf.MultiLayerConfiguration";

    private static final String DATAVEC_GROUPID = "org.datavec";
    private static final String DATAVEC_ARTIFACT = "datavec-api";
    private static final String DATAVEC_CLASS = "org.datavec.api.versioncheck.DataVecApiVersionInfo";

    private static final String ND4J_GROUPID = "org.nd4j";

    private static final String ND4J_JBLAS_CLASS = "org.nd4j.linalg.jblas.JblasBackend";
    private static final String CANOVA_CLASS = "org.canova.api.io.data.DoubleWritable";

    private static final Set<String> GROUPIDS_TO_CHECK = new HashSet<>(Arrays.asList(
            ND4J_GROUPID, DL4J_GROUPID, DATAVEC_GROUPID));    //NOTE: DL4J_GROUPID also covers Arbiter and RL4J

    public enum Detail {
        GAV,
        GAVC,
        FULL
    }

    private VersionCheck(){

    }

    public static void checkVersions(){
        boolean doCheck = Boolean.parseBoolean(System.getProperty(VERSION_CHECK_PROPERTY, "true"));

        if(!doCheck){
            return;
        }

        List<GitRepositoryState> repos = listGitRepositoryInfo();
        Set<String> foundVersions = new HashSet<>();
        for(GitRepositoryState gpr : repos){
            String g = gpr.getGroupId();
            if(g != null && GROUPIDS_TO_CHECK.contains(g)){
                foundVersions.add(g);
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
    }


    public static List<String> listGitPropertiesFiles() {
        Reflections reflections = new Reflections(new ResourcesScanner());

        Set<String> resources = reflections.getResources(Pattern.compile(".*-git.properties"));

        List<String> out = new ArrayList<>(resources);
        Collections.sort(out);      //Equivalent to sorting by groupID and artifactID

        return out;
    }

    public static List<GitRepositoryState> listGitRepositoryInfo() {

        boolean dl4jFound = false;
        boolean datavecFound = false;

        List<GitRepositoryState> repState = new ArrayList<>();
        for(String s : listGitPropertiesFiles()){
            GitRepositoryState grs;

            try{
                grs = new GitRepositoryState(s);
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
                repState.add(new GitRepositoryState(DL4J_GROUPID, DL4J_ARTIFACT, UNKNOWN_VERSION));
            }
        }

        if(!datavecFound){
            //See if pre-0.9.1 DataVec is present on classpath
            if(classExists(DATAVEC_CLASS)){
                repState.add(new GitRepositoryState(DATAVEC_GROUPID, DATAVEC_ARTIFACT, UNKNOWN_VERSION));
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

    public static String versionInfoString() {
        return versionInfoString(Detail.GAV);
    }

    public static String versionInfoString(Detail detail) {
        StringBuilder sb = new StringBuilder();
        for(GitRepositoryState grp : listGitRepositoryInfo()){
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

    public static void logVersionInfo(){
        logVersionInfo(Detail.GAV);
    }

    public static void logVersionInfo(Detail detail){

        List<GitRepositoryState> info = listGitRepositoryInfo();

        for(GitRepositoryState grp : info){
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
