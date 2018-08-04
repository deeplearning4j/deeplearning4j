/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.versioncheck;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.io.FilenameUtils;

import java.io.*;
import java.net.URI;
import java.net.URL;
import java.util.Properties;

/**
 * Created by Alex on 04/08/2017.
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
@Builder
public class VersionInfo {

    private static final int SUFFIX_LENGTH = "-git.properties".length();

    private String groupId;
    private String artifactId;

    private String tags; // =${git.tags} // comma separated tag names
    private String branch; // =${git.branch}
    private String dirty; // =${git.dirty}
    private String remoteOriginUrl; // =${git.remote.origin.url}

    private String commitId; // =${git.commit.id.full} OR ${git.commit.id}
    private String commitIdAbbrev; // =${git.commit.id.abbrev}
    private String describe; // =${git.commit.id.describe}
    private String describeShort; // =${git.commit.id.describe-short}
    private String commitUserName; // =${git.commit.user.opName}
    private String commitUserEmail; // =${git.commit.user.email}
    private String commitMessageFull; // =${git.commit.message.full}
    private String commitMessageShort; // =${git.commit.message.short}
    private String commitTime; // =${git.commit.time}
    private String closestTagName; // =${git.closest.tag.opName}
    private String closestTagCommitCount; // =${git.closest.tag.commit.count}

    private String buildUserName; // =${git.build.user.opName}
    private String buildUserEmail; // =${git.build.user.email}
    private String buildTime; // =${git.build.time}
    private String buildHost; // =${git.build.host}
    private String buildVersion; // =${git.build.version}

    public VersionInfo(String groupId, String artifactId, String buildVersion) {
        this.groupId = groupId;
        this.artifactId = artifactId;
        this.buildVersion = buildVersion;
    }

    public VersionInfo(String propertiesFilePath) throws IOException {
        this(new File(propertiesFilePath).toURI());
    }

    public VersionInfo(URI uri) throws IOException {
        //Can't use new File(uri).getPath() for URIs pointing to resources in JARs
        //But URI.toString() returns "%2520" instead of spaces in path - https://github.com/deeplearning4j/deeplearning4j/issues/6056
        String path = uri.toString().replaceAll("%2520", " ");
        int idxOf = path.lastIndexOf('/');
        idxOf = Math.max(idxOf, path.lastIndexOf('\\'));
        String filename;
        if (idxOf <= 0) {
            filename = path;
        } else {
            filename = path.substring(idxOf + 1);
        }

        idxOf = filename.indexOf('-');
        groupId = filename.substring(0, idxOf);
        artifactId = filename.substring(idxOf + 1, filename.length() - SUFFIX_LENGTH);


        //Extract values from properties file:
        Properties properties = new Properties();
        URL u = new URL(path);  //Can't use URI.toUrl() due to spaces in path
        try (InputStream is = new BufferedInputStream(u.openStream())){ //new FileInputStream(path))) {
            properties.load(is);
        }

        this.tags = String.valueOf(properties.get("git.tags"));
        this.branch = String.valueOf(properties.get("git.branch"));
        this.dirty = String.valueOf(properties.get("git.dirty"));
        this.remoteOriginUrl = String.valueOf(properties.get("git.remote.origin.url"));

        this.commitId = String.valueOf(properties.get("git.commit.id.full")); // OR properties.get("git.commit.id") depending on your configuration
        this.commitIdAbbrev = String.valueOf(properties.get("git.commit.id.abbrev"));
        this.describe = String.valueOf(properties.get("git.commit.id.describe"));
        this.describeShort = String.valueOf(properties.get("git.commit.id.describe-short"));
        this.commitUserName = String.valueOf(properties.get("git.commit.user.name"));
        this.commitUserEmail = String.valueOf(properties.get("git.commit.user.email"));
        this.commitMessageFull = String.valueOf(properties.get("git.commit.message.full"));
        this.commitMessageShort = String.valueOf(properties.get("git.commit.message.short"));
        this.commitTime = String.valueOf(properties.get("git.commit.time"));
        this.closestTagName = String.valueOf(properties.get("git.closest.tag.name"));
        this.closestTagCommitCount = String.valueOf(properties.get("git.closest.tag.commit.count"));

        this.buildUserName = String.valueOf(properties.get("git.build.user.name"));
        this.buildUserEmail = String.valueOf(properties.get("git.build.user.email"));
        this.buildTime = String.valueOf(properties.get("git.build.time"));
        this.buildHost = String.valueOf(properties.get("git.build.host"));
        this.buildVersion = String.valueOf(properties.get("git.build.version"));
    }
}
