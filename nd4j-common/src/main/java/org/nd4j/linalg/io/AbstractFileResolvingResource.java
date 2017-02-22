package org.nd4j.linalg.io;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;


public abstract class AbstractFileResolvingResource extends AbstractResource {
    public AbstractFileResolvingResource() {}

    public File getFile() throws IOException {
        URL url = this.getURL();
        return url.getProtocol().startsWith("vfs")
                        ? AbstractFileResolvingResource.VfsResourceDelegate.getResource(url).getFile()
                        : ResourceUtils.getFile(url, this.getDescription());
    }

    protected File getFileForLastModifiedCheck() throws IOException {
        URL url = this.getURL();
        if (ResourceUtils.isJarURL(url)) {
            URL actualUrl = ResourceUtils.extractJarFileURL(url);
            return actualUrl.getProtocol().startsWith("vfs")
                            ? AbstractFileResolvingResource.VfsResourceDelegate.getResource(actualUrl).getFile()
                            : ResourceUtils.getFile(actualUrl, "Jar URL");
        } else {
            return this.getFile();
        }
    }

    protected File getFile(URI uri) throws IOException {
        return uri.getScheme().startsWith("vfs")
                        ? AbstractFileResolvingResource.VfsResourceDelegate.getResource(uri).getFile()
                        : ResourceUtils.getFile(uri, this.getDescription());
    }

    public boolean exists() {
        try {
            URL ex = this.getURL();
            if (ResourceUtils.isFileURL(ex)) {
                return this.getFile().exists();
            } else {
                URLConnection con = ex.openConnection();
                ResourceUtils.useCachesIfNecessary(con);
                HttpURLConnection httpCon = con instanceof HttpURLConnection ? (HttpURLConnection) con : null;
                if (httpCon != null) {
                    httpCon.setRequestMethod("HEAD");
                    int is = httpCon.getResponseCode();
                    if (is == 200) {
                        return true;
                    }

                    if (is == 404) {
                        return false;
                    }
                }

                if (con.getContentLength() >= 0) {
                    return true;
                } else if (httpCon != null) {
                    httpCon.disconnect();
                    return false;
                } else {
                    InputStream is1 = this.getInputStream();
                    is1.close();
                    return true;
                }
            }
        } catch (IOException var5) {
            return false;
        }
    }

    public boolean isReadable() {
        try {
            URL ex = this.getURL();
            if (!ResourceUtils.isFileURL(ex)) {
                return true;
            } else {
                File file = this.getFile();
                return file.canRead() && !file.isDirectory();
            }
        } catch (IOException var3) {
            return false;
        }
    }

    public long contentLength() throws IOException {
        URL url = this.getURL();
        if (ResourceUtils.isFileURL(url)) {
            return this.getFile().length();
        } else {
            URLConnection con = url.openConnection();
            ResourceUtils.useCachesIfNecessary(con);
            if (con instanceof HttpURLConnection) {
                ((HttpURLConnection) con).setRequestMethod("HEAD");
            }

            return (long) con.getContentLength();
        }
    }

    public long lastModified() throws IOException {
        URL url = this.getURL();
        if (!ResourceUtils.isFileURL(url) && !ResourceUtils.isJarURL(url)) {
            URLConnection con = url.openConnection();
            ResourceUtils.useCachesIfNecessary(con);
            if (con instanceof HttpURLConnection) {
                ((HttpURLConnection) con).setRequestMethod("HEAD");
            }

            return con.getLastModified();
        } else {
            return super.lastModified();
        }
    }

    private static class VfsResourceDelegate {
        private VfsResourceDelegate() {}

        public static Resource getResource(URL url) throws IOException {
            return new VfsResource(VfsUtils.getRoot(url));
        }

        public static Resource getResource(URI uri) throws IOException {
            return new VfsResource(VfsUtils.getRoot(uri));
        }
    }
}
