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

package org.nd4j.linalg.io;

import java.io.File;
import java.io.FileNotFoundException;
import java.net.*;


public abstract class ResourceUtils {
    public static final String CLASSPATH_URL_PREFIX = "classpath:";
    public static final String FILE_URL_PREFIX = "file:";
    public static final String URL_PROTOCOL_FILE = "file";
    public static final String URL_PROTOCOL_JAR = "jar";
    public static final String URL_PROTOCOL_ZIP = "zip";
    public static final String URL_PROTOCOL_VFSZIP = "vfszip";
    public static final String URL_PROTOCOL_VFS = "vfs";
    public static final String URL_PROTOCOL_WSJAR = "wsjar";
    public static final String URL_PROTOCOL_CODE_SOURCE = "code-source";
    public static final String JAR_URL_SEPARATOR = "!/";

    public ResourceUtils() {}

    public static boolean isUrl(String resourceLocation) {
        if (resourceLocation == null) {
            return false;
        } else if (resourceLocation.startsWith("classpath:")) {
            return true;
        } else {
            try {
                new URL(resourceLocation);
                return true;
            } catch (MalformedURLException var2) {
                return false;
            }
        }
    }

    public static URL getURL(String resourceLocation) throws FileNotFoundException {
        Assert.notNull(resourceLocation, "Resource location must not be null");
        if (resourceLocation.startsWith("classpath:")) {
            String ex = resourceLocation.substring("classpath:".length());
            URL ex2 = ClassUtils.getDefaultClassLoader().getResource(ex);
            if (ex2 == null) {
                String description = "class path resource [" + ex + "]";
                throw new FileNotFoundException(description + " cannot be resolved to URL because it does not exist");
            } else {
                return ex2;
            }
        } else {
            try {
                return new URL(resourceLocation);
            } catch (MalformedURLException var5) {
                try {
                    return (new File(resourceLocation)).toURI().toURL();
                } catch (MalformedURLException var4) {
                    throw new FileNotFoundException("Resource location [" + resourceLocation
                                    + "] is neither a URL not a well-formed file path");
                }
            }
        }
    }

    public static File getFile(String resourceLocation) throws FileNotFoundException {
        Assert.notNull(resourceLocation, "Resource location must not be null");
        if (resourceLocation.startsWith("classpath:")) {
            String ex = resourceLocation.substring("classpath:".length());
            String description = "class path resource [" + ex + "]";
            URL url = ClassUtils.getDefaultClassLoader().getResource(ex);
            if (url == null) {
                throw new FileNotFoundException(description + " cannot be resolved to absolute file path "
                                + "because it does not reside in the file system");
            } else {
                return getFile(url, description);
            }
        } else {
            try {
                return getFile(new URL(resourceLocation));
            } catch (MalformedURLException var4) {
                return new File(resourceLocation);
            }
        }
    }

    public static File getFile(URL resourceUrl) throws FileNotFoundException {
        return getFile(resourceUrl, "URL");
    }

    public static File getFile(URL resourceUrl, String description) throws FileNotFoundException {
        Assert.notNull(resourceUrl, "Resource URL must not be null");
        if (!"file".equals(resourceUrl.getProtocol())) {
            throw new FileNotFoundException(description + " cannot be resolved to absolute file path "
                            + "because it does not reside in the file system: " + resourceUrl);
        } else {
            try {
                return new File(toURI(resourceUrl).getSchemeSpecificPart());
            } catch (URISyntaxException var3) {
                return new File(resourceUrl.getFile());
            }
        }
    }

    public static File getFile(URI resourceUri) throws FileNotFoundException {
        return getFile(resourceUri, "URI");
    }

    public static File getFile(URI resourceUri, String description) throws FileNotFoundException {
        Assert.notNull(resourceUri, "Resource URI must not be null");
        if (!"file".equals(resourceUri.getScheme())) {
            throw new FileNotFoundException(description + " cannot be resolved to absolute file path "
                            + "because it does not reside in the file system: " + resourceUri);
        } else {
            return new File(resourceUri.getSchemeSpecificPart());
        }
    }

    public static boolean isFileURL(URL url) {
        String protocol = url.getProtocol();
        return "file".equals(protocol) || protocol.startsWith("vfs");
    }

    public static boolean isJarURL(URL url) {
        String protocol = url.getProtocol();
        return "jar".equals(protocol) || "zip".equals(protocol) || "wsjar".equals(protocol)
                        || "code-source".equals(protocol) && url.getPath().contains("!/");
    }

    public static URL extractJarFileURL(URL jarUrl) throws MalformedURLException {
        String urlFile = jarUrl.getFile();
        int separatorIndex = urlFile.indexOf("!/");
        if (separatorIndex != -1) {
            String jarFile = urlFile.substring(0, separatorIndex);

            try {
                return new URL(jarFile);
            } catch (MalformedURLException var5) {
                if (!jarFile.startsWith("/")) {
                    jarFile = "/" + jarFile;
                }

                return new URL("file:" + jarFile);
            }
        } else {
            return jarUrl;
        }
    }

    public static URI toURI(URL url) throws URISyntaxException {
        return toURI(url.toString());
    }

    public static URI toURI(String location) throws URISyntaxException {
        return new URI(StringUtils.replace(location, " ", "%20"));
    }

    public static void useCachesIfNecessary(URLConnection con) {
        con.setUseCaches(con.getClass().getSimpleName().startsWith("JNLP"));
    }
}
