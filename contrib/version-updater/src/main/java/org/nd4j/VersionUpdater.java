package org.nd4j;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.apache.commons.io.filefilter.RegexFileFilter;
import org.apache.commons.io.filefilter.TrueFileFilter;
import org.nd4j.fileupdater.FileUpdater;
import org.nd4j.fileupdater.impl.CudaFileUpdater;
import org.nd4j.fileupdater.impl.SparkFileUpdater;
import picocli.CommandLine;

import java.io.File;
import java.util.concurrent.Callable;

@CommandLine.Command(name = "version-update")
public class VersionUpdater implements Callable<Integer> {

    @CommandLine.Option(names = {"--root-dir","-d"})
    private File filePath;
    @CommandLine.Option(names = {"--cuda-version","-c"})
    private String newCudaVersion;
    @CommandLine.Option(names = {"--cudnn-version","-cd"})
    private String newCudnnVersion;
    @CommandLine.Option(names = {"--javacpp-version","-jv"})
    private String newJavacppVersion;
    @CommandLine.Option(names = {"--update-type","-t"})
    private String updateType = "cuda";
    private FileUpdater fileUpdater;
    @CommandLine.Option(names = {"--spark-version","-sv"})
    private String sparkVersion;


    public static void main(String... args) {
        CommandLine commandLine = new CommandLine(new VersionUpdater());
        System.exit(commandLine.execute(args));
    }

    @Override
    public Integer call() throws Exception {
        try {
            switch (updateType) {
                case "cuda":
                    fileUpdater = new CudaFileUpdater(newCudaVersion, newJavacppVersion, newCudnnVersion);
                    System.out.println("Updating cuda version using cuda version " + newCudaVersion + " javacpp version " + newJavacppVersion + " cudnn version " + newCudnnVersion);
                    break;
                case "spark":
                    fileUpdater = new SparkFileUpdater(sparkVersion);
                    break;
            }

            for (File f : FileUtils.listFilesAndDirs(filePath, new RegexFileFilter("pom.xml"), new IOFileFilter() {
                @Override
                public boolean accept(File file) {
                    return !file.getName().equals("target");
                }

                @Override
                public boolean accept(File file, String s) {
                    return !file.getName().equals("target");
                }
            })) {
                if (fileUpdater.pathMatches(f)) {
                    fileUpdater.patternReplace(f);
                }
            }
        }catch(Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        return 0;
    }
}
