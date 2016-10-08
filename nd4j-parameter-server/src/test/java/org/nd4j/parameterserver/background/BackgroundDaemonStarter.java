package org.nd4j.parameterserver.background;


import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.parameteraveraging.ParameterAveragingSubscriber;
import org.zeroturnaround.exec.ProcessExecutor;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

/**
 * Start background daemons for tests
 * Credit to:
 * http://stackoverflow.com/questions/636367/executing-a-java-application-in-a-separate-process
 * @author Adam Gibson
 */
@Slf4j
public class BackgroundDaemonStarter {


    /**
     *  Start a slave daemon with the specified master url with the form of:
     *  hostname:port:streamId
     * @param parameterLength the length of the parameters to
     *                        be averaging
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public static int startSlave(int parameterLength,String masterUrl,String mediaDriverDirectory) throws IOException, InterruptedException {
        return exec(ParameterAveragingSubscriber.class,
                mediaDriverDirectory,
                "-l",
                String.valueOf(parameterLength),
                "-p","40126",
                "-h","localhost",
                "-id","10",
                "-pm",masterUrl);
    }

    /**
     *
     * Start a slave daemon with a default url of:
     * localhost:40123:11
     * where the url is:
     * hostname:port:streamId
     * @param parameterLength the parameter length of the ndarrays
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public static int startSlave(int parameterLength,String mediaDriverDirectory) throws IOException, InterruptedException {
        return startSlave(parameterLength,"localhost:40123:11",mediaDriverDirectory);
    }


    public static String slaveConnectionUrl() {
        return "localhost:40126:10";
    }

    /**
     * Master connection url
     * @return
     */
    public static String masterResponderUrl() {
        return "localhost:40124:11";
    }

    /**
     * Master connection url
     * @return
     */
    public static String masterConnectionUrl() {
        return "localhost:40123:11";
    }

    /**
     *
     * @param parameterLength
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public static int startMaster(int parameterLength,String mediaDriverDirectory) throws IOException, InterruptedException {
        return exec(ParameterAveragingSubscriber.class,
                mediaDriverDirectory,
                "-m","true",
                "-l",String.valueOf(parameterLength),
                "-p","40123",
                "-h","localhost",
                "-id","11");
    }

    /**
     * Exec a java process in the background
     * @param klass the main class to run
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public static int exec(Class klass,String mediaDriverDirectory) throws IOException, InterruptedException {
        return exec(klass,null);
    }

    /**
     * Exec a java process in the background
     * @param klass the main class to run
     * @param mediaDriverDirectory the media driver directory to use
     * @param args the args to use (can be null)
     * @return the process exit code
     * @throws IOException
     * @throws InterruptedException
     */
    public static int exec(Class klass,String mediaDriverDirectory,String...args) throws IOException, InterruptedException {
        String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +
                File.separator + "bin" +
                File.separator + "java";
        String classpath = System.getProperty("java.class.path");
        String className = klass.getCanonicalName();
        if(args == null || args.length < 1) {
            try {
                return  new ProcessExecutor().command(javaBin, "-cp", classpath, className)
                          .readOutput(true).redirectOutput(System.out)
                          .redirectError(System.err).execute().getExitValue();
            } catch (TimeoutException e) {
                e.printStackTrace();
            }
        }
        else {
            List<String> args2 = new ArrayList<>(Arrays.asList(javaBin, "-cp", classpath,className,"-md",mediaDriverDirectory));
            args2.addAll(Arrays.asList(args));
            try {
                return  new ProcessExecutor().command(args2)
                        .readOutput(true).redirectOutput(System.out)
                        .redirectError(System.err).execute().getExitValue();
            } catch (TimeoutException e) {
                e.printStackTrace();
            }
        }


        return 1;
    }


}
