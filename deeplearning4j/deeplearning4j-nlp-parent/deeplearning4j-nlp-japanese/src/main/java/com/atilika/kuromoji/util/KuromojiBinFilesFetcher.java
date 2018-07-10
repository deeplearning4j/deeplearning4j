package com.atilika.kuromoji.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by kepricon on 16. 11. 24.
 */
@Slf4j
public class KuromojiBinFilesFetcher {

    private static final String KUROMOJI_RESOURCE_NAME = "kuromoji";

    private KuromojiBinFilesFetcher(){ }

    public static File getKuromojiRoot(){
        return DL4JResources.getDirectory(ResourceType.RESOURCE, KUROMOJI_RESOURCE_NAME);
    }

    public static boolean kuromojiExist() {
        File root = getKuromojiRoot();

        List<File> binFileList = new ArrayList<>();

        binFileList.add(root);
        binFileList.add(new File(root, "characterDefinitions.bin"));
        binFileList.add(new File(root, "connectionCosts.bin"));
        binFileList.add(new File(root, "doubleArrayTrie.bin"));
        binFileList.add(new File(root, "tokenInfoDictionary.bin"));
        binFileList.add(new File(root, "tokenInfoFeaturesMap.bin"));
        binFileList.add(new File(root, "tokenInfoPartOfSpeechMap.bin"));
        binFileList.add(new File(root, "tokenInfoTargetMap.bin"));
        binFileList.add(new File(root, "unknownDictionary.bin"));

        for (File f : binFileList) {
            if (!f.exists()) {
                return false;
            }
        }
        return true;
    }

    public static File downloadAndUntar() throws IOException {
        File rootDir = getKuromojiRoot();
        File[] files = rootDir.listFiles();
        if (rootDir.exists() && files != null && files.length > 0) {
            log.warn("Kuromoji dictionary files exist but failed checks. Deleting and re-downloading.");
            FileUtils.deleteDirectory(rootDir);
            rootDir.mkdir();
        }

        log.info("Downloading Kuromoji bin files...");

        // download kuromoji bin file from azure blob
        File tarFile = new File(rootDir, "kuromoji_bin_files.tar.gz");
        if (!tarFile.isFile()) {
            FileUtils.copyURLToFile(
                            new URL("https://dhkuromoji.blob.core.windows.net/kuromoji/kuromoji_bin_files.tar.gz"),
                            tarFile);
        }
        ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), rootDir.getAbsolutePath());

        return rootDir.getAbsoluteFile();
    }

    @Deprecated
    public static String getRootPath() {
        return getKuromojiRoot().getAbsolutePath();
    }
}
