/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.aws.ec2.provision;

import com.jcraft.jsch.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;

/**
 * Meant for uploading files to remote servers
 * @author Adam Gibson
 *
 */
public class HostProvisioner implements UserInfo {

    private String host;
    private JSch jsch;
    private String user;
    private int port = 22;
    private String password;
    private static final Logger log = LoggerFactory.getLogger(HostProvisioner.class);

    /**
     * 
     * @param host host to connect to (public facing dns)
     * @param user the user to connect with (default root otherwise)
     * @param password the password to use if any
     * @param port the port to connect to(default 22)
     */
    public HostProvisioner(String host, String user, String password, int port) {
        super();
        this.host = host;
        this.user = user;
        this.port = port;
        this.password = password;
        jsch = new JSch();

    }

    /**
     * Connects to port 22
     * @param host host to connect to (public facing dns)
     * @param user the user to connect with (default root otherwise)
     * @param password the password to use if any
     */
    public HostProvisioner(String host, String user, String password) {
        this(host, user, password, 22);
    }

    /**
     * Connects to port 22
     * @param host host to connect to (public facing dns)
     * @param user the user to connect with (default root otherwise)
     */
    public HostProvisioner(String host, String user) {
        this(host, user, "", 22);
    }

    /**
     * Connects to port 22, user root, with no password
     * @param host host to connect to (public facing dns)
     */
    public HostProvisioner(String host) {
        this(host, "root", "", 22);
    }



    public void uploadAndRun(String script, String rootDir) throws Exception {
        String remoteName = rootDir.isEmpty() ? new File(script).getName() : rootDir + "/" + new File(script).getName();
        upload(new File(script), remoteName);

        String remoteCommand = remoteName.charAt(0) != '/' ? "./" + remoteName : remoteName;
        remoteCommand = "chmod +x " + remoteCommand + " && " + remoteCommand;
        runRemoteCommand(remoteCommand);
    }

    public void runRemoteCommand(String remoteCommand) throws Exception {
        Session session = getSession();
        session.connect();
        ChannelExec channel = (ChannelExec) session.openChannel("exec");


        channel.setCommand(remoteCommand);
        channel.setErrStream(System.err);
        channel.setPty(true);

        channel.setOutputStream(System.out);
        channel.connect();
        channel.start();
        InputStream input = channel.getInputStream();

        //start reading the input from the executed commands on the shell
        byte[] tmp = new byte[60000];
        while (true) {
            while (input.available() > 0) {
                int i = input.read(tmp, 0, tmp.length);
                if (i < 0)
                    break;
                log.info(new String(tmp, 0, i));
            }
            if (channel.isClosed()) {
                log.info("exit-status: " + channel.getExitStatus());
                break;
            }

        }

        channel.disconnect();
        session.disconnect();


    }


    private Session getSession() throws Exception {
        Session session = jsch.getSession(user, host, port);
        session.setUserInfo(this);
        return session;
    }

    /**
     * Creates the directory for the file if necessary 
     * and uploads the file
     * @param from the directory to upload from
     * @param to the destination directory on the remote server
     * @throws Exception
     */
    public void uploadForDeployment(String from, String to) throws Exception {
        File fromFile = new File(from);
        if (!to.isEmpty() && fromFile.isDirectory())
            mkDir(to);
        else
            upload(from, to);


    }

    public void addKeyFile(String keyFile) throws Exception {
        jsch.addIdentity(keyFile);
    }

    //creates the directory to upload to
    private void mkDir(String dir) throws Exception {
        Session session = getSession();
        session.connect();
        Channel channel = session.openChannel("sftp");
        channel.connect();

        ChannelSftp c = (ChannelSftp) channel;
        if (!fileExists(dir, c))
            c.mkdir(dir);
        c.exit();
        session.disconnect();
    }

    private boolean fileExists(String dir, ChannelSftp channel) {
        try {
            channel.stat(dir);
            return true;
        } catch (Exception e) {
            return false;
        }
    }


    //uploads the file or listed files in a directory
    private void upload(String fileOrDir, String uploadRootDir) throws Exception {
        if (uploadRootDir.isEmpty())
            uploadRootDir = ".";
        File origin = new File(fileOrDir);

        if (fileOrDir.endsWith(".tar") || fileOrDir.endsWith(".tar.gz")) {
            upload(new File(fileOrDir), uploadRootDir);
            untar(uploadRootDir);
        } else if (origin.isFile()) {
            upload(new File(fileOrDir), uploadRootDir);
        } else {
            File[] childFiles = origin.listFiles();
            if (childFiles != null)
                upload(Arrays.asList(childFiles), uploadRootDir);

        }
    }

    private void untar(String targetRemoteFile) throws Exception {
        this.runRemoteCommand("tar xvf " + targetRemoteFile);
    }

    private void upload(Collection<File> files, String rootDir) throws Exception {
        Session session = getSession();
        session.connect();
        Channel channel = session.openChannel("sftp");
        channel.connect();

        ChannelSftp c = (ChannelSftp) channel;
        for (File f : files) {
            if (f.isDirectory()) {
                log.warn("Skipping " + f.getName());
                continue;
            }

            log.info("Uploading " + f.getName());
            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
            c.put(bis, rootDir + "/" + f.getName());
            bis.close();

        }

        channel.disconnect();
        session.disconnect();


    }

    private void upload(File f, String remoteFile) throws Exception {
        Session session = getSession();
        int numRetries = 0;
        while (numRetries < 3 && !session.isConnected()) {
            try {
                session.connect();
            } catch (Exception e) {
                numRetries++;
            }
        }

        try {
            Channel channel = session.openChannel("sftp");


            channel.connect();

            ChannelSftp c = (ChannelSftp) channel;

            BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
            if (this.fileExists(remoteFile, c))
                if (f.isDirectory())
                    c.rmdir(remoteFile);
                else
                    c.rm(remoteFile);
            c.put(bis, remoteFile);
            bis.close();
            c.exit();
            session.disconnect();
        } catch (Exception e) {
            log.info("Session was down...trying again", e);
            upload(f, remoteFile);
        }
    }



    @Override
    public String getPassphrase() {
        return this.password;
    }


    @Override
    public String getPassword() {
        return this.password;
    }


    @Override
    public boolean promptPassphrase(String arg0) {
        return true;
    }


    @Override
    public boolean promptPassword(String arg0) {
        return true;

    }


    @Override
    public boolean promptYesNo(String arg0) {
        return true;

    }


    @Override
    public void showMessage(String arg0) {
        log.info(arg0);
    }

}
