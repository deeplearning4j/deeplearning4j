package com.atilika.kuromoji.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    protected static final String TEMP_ROOT = System.getProperty("user.home");
    protected static final String KUROMOJI_BIN_ROOT;
    static {
        KUROMOJI_BIN_ROOT = TEMP_ROOT + File.separator + ".dl4j-kuromoji" + File.separator;
    }

    private KuromojiBinFilesFetcher(){ }

    public static boolean kuromojiExist() {
        List<File> binFileList = new ArrayList<>();

        binFileList.add(new File(KUROMOJI_BIN_ROOT));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "characterDefinitions.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "connectionCosts.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "doubleArrayTrie.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "tokenInfoDictionary.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "tokenInfoFeaturesMap.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "tokenInfoPartOfSpeechMap.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "tokenInfoTargetMap.bin"));
        binFileList.add(new File(KUROMOJI_BIN_ROOT, "unknownDictionary.bin"));

        for (File f : binFileList) {
            if (f.exists() == false) {
                return false;
            }
        }
        return true;
    }

    public static File downloadAndUntar() throws IOException {
        File rootFile = new File(KUROMOJI_BIN_ROOT);
        if (rootFile.exists()) {
            throw new IOException("remove exist directory : " + KUROMOJI_BIN_ROOT);
        } else {
            rootFile.mkdir();

            log.info("Downloading Kuromoji bin files...");

            // download kuromoji bin file from azure blob
            File tarFile = new File(KUROMOJI_BIN_ROOT, "kuromoji_bin_files.tar.gz");
            if (!tarFile.isFile()) {
                FileUtils.copyURLToFile(
                                new URL("https://dhkuromoji.blob.core.windows.net/kuromoji/kuromoji_bin_files.tar.gz"),
                                tarFile);
            }
            ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), rootFile.getAbsolutePath());
        }

        return null;
    }

    public static String getRootPath() {
        return KUROMOJI_BIN_ROOT;
    }
}
