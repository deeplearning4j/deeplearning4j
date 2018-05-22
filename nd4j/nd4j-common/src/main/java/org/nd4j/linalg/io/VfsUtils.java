package org.nd4j.linalg.io;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URL;



public abstract class VfsUtils {
    private static final Logger logger = LoggerFactory.getLogger(VfsUtils.class);
    private static final String VFS2_PKG = "org.jboss.virtual.";
    private static final String VFS3_PKG = "org.jboss.vfs.";
    private static final String VFS_NAME = "VFS";
    private static VfsUtils.VFS_VER version;
    private static Method VFS_METHOD_GET_ROOT_URL = null;
    private static Method VFS_METHOD_GET_ROOT_URI = null;
    private static Method VIRTUAL_FILE_METHOD_EXISTS = null;
    private static Method VIRTUAL_FILE_METHOD_GET_INPUT_STREAM;
    private static Method VIRTUAL_FILE_METHOD_GET_SIZE;
    private static Method VIRTUAL_FILE_METHOD_GET_LAST_MODIFIED;
    private static Method VIRTUAL_FILE_METHOD_TO_URL;
    private static Method VIRTUAL_FILE_METHOD_TO_URI;
    private static Method VIRTUAL_FILE_METHOD_GET_NAME;
    private static Method VIRTUAL_FILE_METHOD_GET_PATH_NAME;
    private static Method VIRTUAL_FILE_METHOD_GET_CHILD;
    protected static Class<?> VIRTUAL_FILE_VISITOR_INTERFACE;
    protected static Method VIRTUAL_FILE_METHOD_VISIT;
    private static Method VFS_UTILS_METHOD_IS_NESTED_FILE = null;
    private static Method VFS_UTILS_METHOD_GET_COMPATIBLE_URI = null;
    private static Field VISITOR_ATTRIBUTES_FIELD_RECURSE = null;
    private static Method GET_PHYSICAL_FILE = null;

    public VfsUtils() {}

    protected static Object invokeVfsMethod(Method method, Object target, Object... args) throws IOException {
        try {
            return method.invoke(target, args);
        } catch (InvocationTargetException var5) {
            Throwable targetEx = var5.getTargetException();
            if (targetEx instanceof IOException) {
                throw (IOException) targetEx;
            }

            ReflectionUtils.handleInvocationTargetException(var5);
        } catch (Exception var6) {
            ReflectionUtils.handleReflectionException(var6);
        }

        throw new IllegalStateException("Invalid code path reached");
    }

    static boolean exists(Object vfsResource) {
        try {
            return ((Boolean) invokeVfsMethod(VIRTUAL_FILE_METHOD_EXISTS, vfsResource, new Object[0])).booleanValue();
        } catch (IOException var2) {
            return false;
        }
    }

    static boolean isReadable(Object vfsResource) {
        try {
            return ((Long) invokeVfsMethod(VIRTUAL_FILE_METHOD_GET_SIZE, vfsResource, new Object[0])).longValue() > 0L;
        } catch (IOException var2) {
            return false;
        }
    }

    static long getSize(Object vfsResource) throws IOException {
        return ((Long) invokeVfsMethod(VIRTUAL_FILE_METHOD_GET_SIZE, vfsResource, new Object[0])).longValue();
    }

    static long getLastModified(Object vfsResource) throws IOException {
        return ((Long) invokeVfsMethod(VIRTUAL_FILE_METHOD_GET_LAST_MODIFIED, vfsResource, new Object[0])).longValue();
    }

    static InputStream getInputStream(Object vfsResource) throws IOException {
        return (InputStream) invokeVfsMethod(VIRTUAL_FILE_METHOD_GET_INPUT_STREAM, vfsResource, new Object[0]);
    }

    static URL getURL(Object vfsResource) throws IOException {
        return (URL) invokeVfsMethod(VIRTUAL_FILE_METHOD_TO_URL, vfsResource, new Object[0]);
    }

    static URI getURI(Object vfsResource) throws IOException {
        return (URI) invokeVfsMethod(VIRTUAL_FILE_METHOD_TO_URI, vfsResource, new Object[0]);
    }

    static String getName(Object vfsResource) {
        try {
            return (String) invokeVfsMethod(VIRTUAL_FILE_METHOD_GET_NAME, vfsResource, new Object[0]);
        } catch (IOException var2) {
            throw new IllegalStateException("Cannot get resource name", var2);
        }
    }

    static Object getRelative(URL url) throws IOException {
        return invokeVfsMethod(VFS_METHOD_GET_ROOT_URL, null, new Object[] {url});
    }

    static Object getChild(Object vfsResource, String path) throws IOException {
        return invokeVfsMethod(VIRTUAL_FILE_METHOD_GET_CHILD, vfsResource, new Object[] {path});
    }

    static File getFile(Object vfsResource) throws IOException {
        if (VfsUtils.VFS_VER.V2.equals(version)) {
            if (((Boolean) invokeVfsMethod(VFS_UTILS_METHOD_IS_NESTED_FILE, null, new Object[] {vfsResource}))
                            .booleanValue()) {
                throw new IOException("File resolution not supported for nested resource: " + vfsResource);
            } else {
                try {
                    return new File((URI) invokeVfsMethod(VFS_UTILS_METHOD_GET_COMPATIBLE_URI, null,
                                    new Object[] {vfsResource}));
                } catch (Exception var2) {
                    throw new IOException("Failed to obtain File reference for " + vfsResource, var2);
                }
            }
        } else {
            return (File) invokeVfsMethod(GET_PHYSICAL_FILE, vfsResource, new Object[0]);
        }
    }

    static Object getRoot(URI url) throws IOException {
        return invokeVfsMethod(VFS_METHOD_GET_ROOT_URI, null, new Object[] {url});
    }

    protected static Object getRoot(URL url) throws IOException {
        return invokeVfsMethod(VFS_METHOD_GET_ROOT_URL, null, new Object[] {url});
    }

    protected static Object doGetVisitorAttribute() {
        return ReflectionUtils.getField(VISITOR_ATTRIBUTES_FIELD_RECURSE, null);
    }

    protected static String doGetPath(Object resource) {
        return (String) ReflectionUtils.invokeMethod(VIRTUAL_FILE_METHOD_GET_PATH_NAME, resource);
    }

    static {
        ClassLoader loader = VfsUtils.class.getClassLoader();

        String pkg;
        Class vfsClass;
        try {
            vfsClass = loader.loadClass("org.jboss.vfs.VFS");
            version = VfsUtils.VFS_VER.V3;
            pkg = "org.jboss.vfs.";
            if (logger.isDebugEnabled()) {
                logger.debug("JBoss VFS packages for JBoss AS 6 found");
            }
        } catch (ClassNotFoundException var9) {
            if (logger.isDebugEnabled()) {
                logger.debug("JBoss VFS packages for JBoss AS 6 not found; falling back to JBoss AS 5 packages");
            }

            try {
                vfsClass = loader.loadClass("org.jboss.virtual.VFS");
                version = VfsUtils.VFS_VER.V2;
                pkg = "org.jboss.virtual.";
                if (logger.isDebugEnabled()) {
                    logger.debug("JBoss VFS packages for JBoss AS 5 found");
                }
            } catch (ClassNotFoundException var8) {
                logger.error("JBoss VFS packages (for both JBoss AS 5 and 6) were not found - JBoss VFS support disabled");
                throw new IllegalStateException("Cannot detect JBoss VFS packages", var8);
            }
        }

        try {
            String ex = VfsUtils.VFS_VER.V3.equals(version) ? "getChild" : "getRoot";
            VFS_METHOD_GET_ROOT_URL = ReflectionUtils.findMethod(vfsClass, ex, new Class[] {URL.class});
            VFS_METHOD_GET_ROOT_URI = ReflectionUtils.findMethod(vfsClass, ex, new Class[] {URI.class});
            Class virtualFile = loader.loadClass(pkg + "VirtualFile");
            VIRTUAL_FILE_METHOD_EXISTS = ReflectionUtils.findMethod(virtualFile, "exists");
            VIRTUAL_FILE_METHOD_GET_INPUT_STREAM = ReflectionUtils.findMethod(virtualFile, "openStream");
            VIRTUAL_FILE_METHOD_GET_SIZE = ReflectionUtils.findMethod(virtualFile, "getSize");
            VIRTUAL_FILE_METHOD_GET_LAST_MODIFIED = ReflectionUtils.findMethod(virtualFile, "getLastModified");
            VIRTUAL_FILE_METHOD_TO_URI = ReflectionUtils.findMethod(virtualFile, "toURI");
            VIRTUAL_FILE_METHOD_TO_URL = ReflectionUtils.findMethod(virtualFile, "toURL");
            VIRTUAL_FILE_METHOD_GET_NAME = ReflectionUtils.findMethod(virtualFile, "getName");
            VIRTUAL_FILE_METHOD_GET_PATH_NAME = ReflectionUtils.findMethod(virtualFile, "getPathName");
            GET_PHYSICAL_FILE = ReflectionUtils.findMethod(virtualFile, "getPhysicalFile");
            ex = VfsUtils.VFS_VER.V3.equals(version) ? "getChild" : "findChild";
            VIRTUAL_FILE_METHOD_GET_CHILD = ReflectionUtils.findMethod(virtualFile, ex, new Class[] {String.class});
            Class utilsClass = loader.loadClass(pkg + "VFSUtils");
            VFS_UTILS_METHOD_GET_COMPATIBLE_URI =
                            ReflectionUtils.findMethod(utilsClass, "getCompatibleURI", new Class[] {virtualFile});
            VFS_UTILS_METHOD_IS_NESTED_FILE =
                            ReflectionUtils.findMethod(utilsClass, "isNestedFile", new Class[] {virtualFile});
            VIRTUAL_FILE_VISITOR_INTERFACE = loader.loadClass(pkg + "VirtualFileVisitor");
            VIRTUAL_FILE_METHOD_VISIT = ReflectionUtils.findMethod(virtualFile, "visit",
                            new Class[] {VIRTUAL_FILE_VISITOR_INTERFACE});
            Class visitorAttributesClass = loader.loadClass(pkg + "VisitorAttributes");
            VISITOR_ATTRIBUTES_FIELD_RECURSE = ReflectionUtils.findField(visitorAttributesClass, "RECURSE");
        } catch (ClassNotFoundException var7) {
            throw new IllegalStateException("Could not detect the JBoss VFS infrastructure", var7);
        }
    }

    private static enum VFS_VER {
        V2, V3;

        private VFS_VER() {}
    }
}
