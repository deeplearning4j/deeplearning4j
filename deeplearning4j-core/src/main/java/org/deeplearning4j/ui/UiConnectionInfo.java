package org.deeplearning4j.ui;

import lombok.Data;
import lombok.NonNull;

import java.util.Random;

/**
 * POJO describing the location and credentials for DL4j UiServer instance
 *
 * @author raver119@gmail.com
 */
@Data
public class UiConnectionInfo {
    private String sessionId;
    private String login;
    private String password;
    private String address = "localhost";
    private int port = 8080;
    private String path = "";
    private boolean useHttps;

    public UiConnectionInfo() {
        this.sessionId = java.util.UUID.randomUUID().toString();
    }

    public void setSessionId(@NonNull String sessionId) {
        this.sessionId = sessionId;
    }

    /**
     * This method returns scheme, address and port for this UiConnectionInfo
     *
     * i.e: https://localhost:8080
     *
     * @return
     */
    public String getFirstPart() {
        StringBuilder builder = new StringBuilder();

        builder
                .append(useHttps ? "https" : "http").append("://")
                .append(address).append(":")
                .append(port).append("");

        return builder.toString();
    }

    public String getSecondPart() {
        return getSecondPart("");
    }

    public String getSecondPart(String nPath) {
        StringBuilder builder = new StringBuilder();

        if (path != null && !path.isEmpty()) {
            builder.append(path.startsWith("/") ? path : ("/" + path)).append("/");
        }

        if (nPath != null) {
            nPath = nPath.replaceFirst("^/", "");
            builder.append(path.endsWith("/") ? nPath : ("/" + nPath)).append("/");
        }


        return builder.toString().replaceAll("\\/{2,}","/");
    }

    public String getFullAddress(String nPath) {
        if (nPath == null || nPath.isEmpty()) {
            return getFullAddress();
        } else {
            return getFirstPart() + getSecondPart(nPath) + "?sid=" + this.getSessionId();
        }
    }

    public String getFullAddress() {
        return getFirstPart() + getSecondPart();
    }

    public static class Builder {
        private UiConnectionInfo info = new UiConnectionInfo();

        /**
         * This method allows you to specify sessionId for this UiConnectionInfo instance
         *
         * PLEASE NOTE: This is not recommended. Advised behaviour - keep it random, as is.
         *
         * @param sessionId
         * @return
         */
        public Builder setSessionId(@NonNull String sessionId) {
            info.setSessionId(sessionId);
            return this;
        }

        public Builder setLogin(@NonNull String login) {
            info.setLogin(login);
            return this;
        }

        public Builder setPassword(String password) {
            info.setPassword(password);
            return this;
        }

        public Builder setAddress(@NonNull String address) {
            info.setAddress(address);
            return this;
        }

        public Builder setPort(int port) {
            if (port <= 0) throw new IllegalStateException("UiServer port can't be <= 0");
            info.setPort(port);
            return this;
        }

        public Builder enableHttps(boolean reallyEnable) {
            info.setUseHttps(reallyEnable);
            return this;
        }

        /**
         * If you're using UiServer as servlet, located not at root folder of webserver (i.e. http://yourdomain.com/somepath/webui/), you can set path here.
         * For provided example path will be "/somepath/webui/"
         *
         * @param path
         * @return
         */
        public Builder setPath(String path) {
            info.setPath(path);
            return this;
        }

        public UiConnectionInfo build() {
            return info;
        }
    }
}
