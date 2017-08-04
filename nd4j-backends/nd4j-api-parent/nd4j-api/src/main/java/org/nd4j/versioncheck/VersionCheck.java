package org.nd4j.versioncheck;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.*;

/**
 * Created by Alex on 04/08/2017.
 */
@Slf4j
public class VersionCheck {

    public enum Detail {
        GAV,
        GAVC,
        FULL
    }

    private VersionCheck(){

    }

    public static List<String> listGitPropertiesFiles() throws IOException {

        List<String> files = IOUtils.readLines(VersionCheck.class.getClassLoader()
                .getResourceAsStream("ai/skymind/"), Charsets.UTF_8);

        if(files == null || files.size() == 0){
            return Collections.emptyList();
        }

        List<String> out = new ArrayList<>();
        for(String s : files){
            if(!s.endsWith("-git.properties")){
                continue;
            }
            out.add("ai/skymind/" + s);
        }

        //Sort by file path - should be equivalent to sorting by groupId then artifactId
        Collections.sort(out);

        return out;
    }

    public static List<GitRepositoryState> listGitRepositoryInfo() throws IOException {
        List<GitRepositoryState> repState = new ArrayList<>();
        for(String s : listGitPropertiesFiles()){
            repState.add(new GitRepositoryState(s));
        }

        return repState;
    }

    public static void logVersionInfo() throws IOException {
        logVersionInfo(Detail.GAV);
    }

    public static void logVersionInfo(Detail detail) throws IOException{

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

    public static GitRepositoryState getGitRepositoryState(String propertiesFilePath) throws IOException {
        return new GitRepositoryState(propertiesFilePath);
    }

    public static void main(String[] args) throws Exception {

        logVersionInfo(Detail.GAV);


    }

//    private static List<String>

}
