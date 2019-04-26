################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
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
from .jarmgr import *
from .jarmgr import _MY_DIR
from .pom import *
from .docker import docker_file
import platform
import os
import warnings
import os
from subprocess import call as py_call
import json


def call(arglist):
    error = py_call(arglist)
    if error:
        raise Exception('Subprocess error for command: ' + str(arglist))


_CONFIG_FILE = os.path.join(_MY_DIR, 'config.json')


# Default config
_CONFIG = {
    'dl4j_version': '1.0.0-SNAPSHOT',
    'dl4j_core': True,
    'datavec': True,
    'spark': True,
    'spark_version': '2',
    'scala_version': '2.11',
    'nd4j_backend': 'cpu',
    'validate_jars': True
}


def _is_sub_set(config1, config2):
    # check if config1 is a subset of config2
    # if config1 < config2, then we can use config2 jar
    # for config1 as well
    if config1['dl4j_version'] != config1['dl4j_version']:
        return False
    if config1['dl4j_core'] > config2['dl4j_core']:
        return False
    if config1['nd4j_backend'] != config2['nd4j_backend']:
        return False
    if config1['datavec']:
        if not config2['datavec']:
            return False
        if config1['spark'] > config2['spark']:
            return False
        if config1['spark_version'] != config2['spark_version']:
            return False
        if config1['scala_version'] != config2['scala_version']:
            return False
    return True


def _write_config(filepath=None):
    if not filepath:
        filepath = _CONFIG_FILE
    with open(filepath, 'w') as f:
        json.dump(_CONFIG, f)


if os.path.isfile(_CONFIG_FILE):
    with open(_CONFIG_FILE, 'r') as f:
        _CONFIG.update(json.load(f))
else:
    _write_config()


def set_config(config):
    _CONFIG.update(config)
    _write_config()


def get_config():
    return _CONFIG


def validate_config(config=None):
    if config is None:
        config = _CONFIG
    valid_options = {
        'spark_version': ['1', '2'],
        'scala_version': ['2.10', '2.11'],
        'nd4j_backend': ['cpu', 'gpu']
    }
    for k, vs in valid_options.items():
        v = config.get(k)
        if v is None:
            raise KeyError('Key not found in config : {}.'.format(k))
        if v not in vs:
            raise ValueError(
                'Invalid value {} for key {} in config. Valid values are: {}.'.format(v, k, vs))

    # spark 2 does not work with scala 2.10
    if config['spark_version'] == '2' and config['scala_version'] == '2.10':
        raise ValueError(
            'Scala 2.10 does not work with spark 2. Set scala_version to 2.11 in pydl4j config. ')


def _get_context_from_config(config=None):
    if not config:
        config = _CONFIG
    # e.g pydl4j-1.0.0-SNAPSHOT-cpu-core-datavec-spark2-2.11

    context = 'pydl4j-{}'.format(config['dl4j_version'])
    context += '-' + config['nd4j_backend']
    if config['dl4j_core']:
        context += '-core'
    if config['datavec']:
        context += '-datavec'
        if config['spark']:
            spark_version = config['spark_version']
            scala_version = config['scala_version']
            context += '-spark' + spark_version + '-' + scala_version
    return context


def _get_config_from_context(context):
    config = {}
    backends = ['cpu', 'gpu']
    for b in backends:
        if '-' + b in context:
            config['nd4j_backend'] = b
            config['dl4j_version'] = context.split('-' + b)[0][len('pydl4j-'):]
            break
    config['dl4j_core'] = '-core' in context
    set_defs = False
    if '-datavec' in context:
        config['datavec'] = True
        if '-spark' in context:
            config['spark'] = True
            sp_sc_ver = context.split('-spark')[1]
            sp_ver, sc_ver = sp_sc_ver.split('-')
            config['spark_version'] = sp_ver
            config['scala_version'] = sc_ver
        else:
            config['spark'] = False
            set_defs = True
    else:
        config['datavec'] = False
        set_defs = True
    if set_defs:
        config['spark_version'] = '2'
        config['scala_version'] = '2.11'
    validate_config(config)
    return config


set_context(_get_context_from_config())


def create_pom_from_config():
    config = get_config()
    pom = pom_template()
    dl4j_version = config['dl4j_version']
    nd4j_backend = config['nd4j_backend']
    use_spark = config['spark']
    scala_version = config['scala_version']
    spark_version = config['spark_version']
    use_dl4j_core = config['dl4j_core']
    use_datavec = config['datavec']

    datavec_deps = datavec_dependencies() if use_datavec else ""
    pom = pom.replace('{datavec.dependencies}', datavec_deps)

    core_deps = dl4j_core_dependencies() if use_dl4j_core else ""
    pom = pom.replace('{dl4j.core.dependencies}', core_deps)

    spark_deps = spark_dependencies() if use_spark else ""
    pom = pom.replace('{spark.dependencies}', spark_deps)

    pom = pom.replace('{dl4j.version}', dl4j_version)

    if nd4j_backend == 'cpu':
        platform_backend = "nd4j-native-platform"
        backend = "nd4j-native"
    else:
        platform_backend = "nd4j-cuda-9.2-platform"
        platform_backend = "nd4j-cuda-9.2"

    pom = pom.replace('{nd4j.backend}', backend)
    pom = pom.replace('{nd4j.platform.backend}', platform_backend)

    if use_spark:
        pom = pom.replace('{scala.binary.version}', scala_version)
        # this naming convention seems a little off
        if "SNAPSHOT" in dl4j_version:
            dl4j_version = dl4j_version.replace("-SNAPSHOT", "")
            dl4j_spark_version = dl4j_version + "_spark_" + spark_version + "-SNAPSHOT"
        else:
            dl4j_spark_version = dl4j_version + "_spark_" + spark_version
        pom = pom.replace('{dl4j.spark.version}', dl4j_spark_version)

    # TODO replace if exists
    pom_xml = os.path.join(_MY_DIR, 'pom.xml')
    with open(pom_xml, 'w') as pom_file:
        pom_file.write(pom)


def docker_build():
    docker_path = os.path.join(_MY_DIR, 'Dockerfile')
    docker_string = docker_file()
    with open(docker_path, 'w') as f:
        f.write(docker_string)

    call(["docker", "build", _MY_DIR, "-t", "pydl4j"])


def docker_run():
    create_pom_from_config()
    py_call(["docker", "run", "--mount", "src=" +
             _MY_DIR + ",target=/app,type=bind", "pydl4j"])
    # docker will build into <context>/target, need to move to context dir
    context_dir = get_dir()
    config = get_config()
    dl4j_version = config['dl4j_version']
    jar_name = "pydl4j-{}-bin.jar".format(dl4j_version)
    base_target_dir = os.path.join(_MY_DIR, "target")
    source = os.path.join(base_target_dir, jar_name)
    target = os.path.join(context_dir, jar_name)
    _write_config(os.path.join(context_dir, 'config.json'))
    if os.path.isfile(target):
        os.remove(target)
    os.rename(source, target)


def is_docker_available():
    devnull = open(os.devnull, 'w')
    try:
        py_call(["docker", "--help"], stdout=devnull, stderr=devnull)
        return True
    except Exception:
        return False


def _maven_build(use_docker):
    if use_docker:
        docker_build()
        docker_run()
    else:
        create_pom_from_config()
        pom_xml = os.path.join(_MY_DIR, 'pom.xml')
        command = 'mvn clean install -f ' + pom_xml
        os.system(command)
        version = _CONFIG['dl4j_version']
        jar_name = "pydl4j-{}-bin.jar".format(version)
        source = os.path.join(_MY_DIR, 'target', jar_name)
        target = os.path.join(get_dir(), jar_name)
        if os.path.isfile(target):
            os.remove(target)
        os.rename(source, target)


def maven_build():
    if is_docker_available():
        print("Docker available. Starting build...")
        _maven_build(use_docker=True)
    else:
        warnings.warn(
            "Docker unavailable. Attempting alternate implementation.")
        _maven_build(use_docker=False)


def validate_jars():
    if not _CONFIG['validate_jars']:
        return
    # builds jar if not available for given context
    jars = get_jars()
    dl4j_version = _CONFIG['dl4j_version']
    jar = "pydl4j-{}-bin.jar".format(dl4j_version)
    if jar not in jars:
        # jar not found
        # but its possible a jar exists in a different
        # context. If that context is a "super set" of
        # of the current one, we can use its jar!
        original_context = context()
        contexts = _get_all_contexts()
        found_super_set_jar = False
        for c in contexts:
            config = _get_config_from_context(c)
            if _is_sub_set(_CONFIG, config):
                set_context(c)
                jars = get_jars()
                if jar in jars:
                    found_super_set_jar = True
                    break
        if not found_super_set_jar:
            set_context(original_context)
            print("pdl4j: required uberjar not found, building with docker...")
            maven_build()


def validate_nd4j_jars():
    validate_jars()


def validate_datavec_jars():
    if not _CONFIG['datavec']:
        _CONFIG['datavec'] = True
        _write_config()
        context = _get_context_from_config()
        set_context(context)
    validate_jars()


def _get_all_contexts():
    c = os.listdir(_MY_DIR)
    return [x for x in c if x.startswith('pydl4j')]


def set_jnius_config():
    try:
        import jnius_config
        path = get_dir()
        if path[-1] == '*':
            jnius_config.add_classpath(path)
        elif os.path.isfile(path):
            jnius_config.add_classpath(path)
        else:
            path = os.path.join(path, '*')
            jnius_config.add_classpath(path)
    # Further options can be set by individual projects
    except ImportError:
        warnings.warn('Pyjnius not installed.')


def add_classpath(path):
    try:
        import jnius_config
        jnius_config.add_classpath(path)
    except ImportError:
        warnings.warn('Pyjnius not installed.')


set_jnius_config()
