package org.nd4j.versioncheck;

import com.google.common.collect.Multimap;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.reflections.Reflections;
import org.reflections.scanners.AbstractScanner;
import org.reflections.util.ConfigurationBuilder;
import org.reflections.util.FilterBuilder;
import org.reflections.scanners.ResourcesScanner;
import org.reflections.vfs.Vfs;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
        Reflections reflections = new Reflections(new ResourcesScanner());

        Set<String> resources = reflections.getResources(Pattern.compile(".*-git.properties"));

        return new ArrayList<>(resources);
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

    public static String versionInfoString() throws IOException {
        return versionInfoString(Detail.GAV);
    }

    public static String versionInfoString(Detail detail) throws IOException {
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

}
