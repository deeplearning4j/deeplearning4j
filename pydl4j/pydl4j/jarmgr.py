#  /* ******************************************************************************
#   * Copyright (c) 2021 Deeplearning4j Contributors
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

################################################################################
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

from .downloader import download as download_file
from .mvn import *
import requests
import json
import os


def mkdir(x):
    if not os.path.isdir(x):
        os.mkdir(x)


_CONTEXT_NAME = None
_CONTEXT_DIR = None
_USER_PATH = os.path.expanduser('~')
_DL4J_DIR = os.path.join(_USER_PATH, '.deeplearning4j')
mkdir(_DL4J_DIR)
_MY_DIR = os.path.join(_DL4J_DIR, 'pydl4j')
mkdir(_MY_DIR)

_URLS_FILE = os.path.join(_MY_DIR, 'urls.json')
if os.path.isfile(_URLS_FILE):
    with open(_URLS_FILE, 'r') as f:
        _URLS = json.load(f)
else:
    _URLS = {}


def _write_urls():
    with open(_URLS_FILE, 'w') as f:
        json.dump(_URLS, f)


_cache = {}


def _read(url):
    text = _cache.get(url)
    if text is None:
        text = requests.get(url).text
        if not text:
            raise Exception('Empty response. Check connectivity.')
        _cache[url] = text
    return text


def _parse_contents(text):
    contents = text.split('<pre id="contents">')[1]
    contents = contents.split('</pre>')[0]
    contents = contents.split('<a href="')
    _ = contents.pop(0)
    link_to_parent = contents.pop(0)
    contents = list(map(lambda x: x.split('"')[0], contents))
    contents = [c[:-1]
                for c in contents if c[-1] == '/']  # removes meta data files
    return contents


def check(f):
    def wrapper(*args, **kwargs):
        if _CONTEXT_NAME is None:
            raise Exception(
                'Context not set! Set context using pydl4j.set_context()')
        mkdir(_CONTEXT_DIR)
        return f(*args, **kwargs)
    return wrapper


def set_context(name):
    global _CONTEXT_NAME
    global _CONTEXT_DIR
    _CONTEXT_NAME = name
    if name is None:
        _CONTEXT_DIR = None
    else:
        _CONTEXT_DIR = os.path.join(_MY_DIR, name)
        mkdir(_CONTEXT_DIR)


@check
def context():
    return _CONTEXT_NAME


@check
def get_dir():
    path = os.environ.get('PYDL4J_CLASS_PATH')
    if path is None:
        # For backward compatibility
        path = os.environ.get('JUMPY_CLASS_PATH')
    if path is None:
        return _CONTEXT_DIR
    return path


@check
def install(url, jar_name=None):
    if not jar_name:
        jar_name = os.path.basename(url)
    if os.path.isfile(url):
        command = 'cp {} {}'.format(url, os.path.join(get_dir(), jar_name))
        if os.name != 'nt':
            command = 'sudo ' + command
        os.system(command)
        return
    jar_path = os.path.join(_CONTEXT_DIR, jar_name)
    temp_jar_path = jar_path + '.tmp'
    if os.path.isfile(temp_jar_path):
        os.remove(temp_jar_path)
    download_file(url, temp_jar_path)
    if os.path.isfile(jar_path):
        os.remove(jar_path)
    os.rename(temp_jar_path, jar_path)
    if _CONTEXT_NAME not in _URLS:
        _URLS[_CONTEXT_NAME] = {jar_name: url}
    else:
        _URLS[_CONTEXT_NAME][jar_name] = url
    _write_urls()


@check
def mvn_install(group, artifact, version=None):
    if version is None:
        version = get_latest_version(group, artifact)
        print('Version not specified for org.{}.{}.'
              'Installing latest version: {}.'.format(group, artifact, version))
    url = get_jar_url(group, artifact, version)
    install(url)


@check
def uninstall(artifact, version=None):
    files = os.listdir(_CONTEXT_DIR)
    if version is not None:
        artifact += '-' + version
    if not artifact.endswith('.jar'):
        artifact += '.jar'
    found = False
    for f in files:
        if f == artifact:  # could have wildcards, disabled for now
            os.remove(os.path.join(_CONTEXT_DIR, f))
            if _CONTEXT_NAME in _URLS:
                if f in _URLS[_CONTEXT_NAME]:
                    _URLS[_CONTEXT_NAME].pop(f)
                    _write_urls()
            found = True
            break
    if not found:
        raise Exception('No matching jars found : {}. '
                        'Use pydl4j.get_jars() to see available jars.'.format(artifact))


@check
def get_jars():
    return [x for x in os.listdir(_CONTEXT_DIR) if x.endswith('.jar')]


@check
def clear_context():
    for j in get_jars():
        uninstall(j)
    try:
        os.remove(_CONTEXT_DIR)
    except:
        pass


def update(jar):
    install(_URLS[_CONTEXT_NAME][jar])
