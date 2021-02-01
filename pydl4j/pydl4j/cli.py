#!/usr/bin python

# -*- coding: utf-8 -*-

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

import argparse
import json
import os
import sys
import pkg_resources
import argcomplete
import traceback
import subprocess
import click
from click.exceptions import ClickException
from dateutil import parser

from .pydl4j import set_config, get_config
from .pydl4j import validate_config, is_docker_available
from .pydl4j import _maven_build


if sys.version_info[0] == 2:
    input = raw_input


_CONFIG = get_config()

DEFAULT_DL4J_VERSION = _CONFIG['dl4j_version']
DEFAULT_BACKEND = _CONFIG['nd4j_backend']
DEFAULT_DATAVEC = _CONFIG['datavec']
DEFAULT_SPARK = _CONFIG['spark']
DEFAULT_SPARK_MAJOR = _CONFIG['spark_version']
DEFAULT_SCALA_VERSION = _CONFIG['scala_version']
DEFAULT_SPARK_DETAILS = 'y'


def to_bool(string):
    if type(string) is bool:
        return string
    return True if string[0] in ["Y", "y"] else False


class CLI(object):

    def __init__(self):
        self.var_args = None
        self.command = None

    def command_dispatcher(self, args=None):
        desc = ('pydl4j,  a system to manage your DL4J dependencies from Python.\n')
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument(
            '-v', '--version', action='version',
            version=pkg_resources.get_distribution("pydl4j").version,
            help='Print pydl4j version'
        )

        subparsers = parser.add_subparsers(title='subcommands', dest='command')
        subparsers.add_parser('init', help='Initialize pydl4j')
        subparsers.add_parser('install', help='Install jars for pydl4j')

        argcomplete.autocomplete(parser)
        args = parser.parse_args(args)
        self.var_args = vars(args)

        if not args.command:
            parser.print_help()
            return

        self.command = args.command

        if self.command == 'init':
            self.init()
            return

        if self.command == 'install':
            self.install()
            return

    def init(self):

        click.echo(click.style(u"""\n██████╗ ██╗   ██╗██████╗ ██╗██╗  ██╗     ██╗
██╔══██╗╚██╗ ██╔╝██╔══██╗██║██║  ██║     ██║
██████╔╝ ╚████╔╝ ██║  ██║██║███████║     ██║
██╔═══╝   ╚██╔╝  ██║  ██║██║╚════██║██   ██║
██║        ██║   ██████╔╝███████╗██║╚█████╔╝
╚═╝        ╚═╝   ╚═════╝ ╚══════╝╚═╝ ╚════╝ \n""", fg='blue', bold=True))

        click.echo(click.style("pydl4j", bold=True) +
                   " is a system to manage your DL4J dependencies from Python!\n")

        # DL4J version
        dl4j_version = input("Which DL4J version do you want to use for your Python projects? (default '%s'): " %
                             DEFAULT_DL4J_VERSION) or DEFAULT_DL4J_VERSION
        # TODO: check if input is valid

        # ND4J backend
        backend = input("Which backend would you like to use ('cpu' or 'gpu')? (default '%s'): " %
                        DEFAULT_BACKEND) or DEFAULT_BACKEND
        backend = backend.lower()

        # DataVec usage
        datavec = input(
            "Do you need DL4J DataVec for ETL? (default 'y') [y/n]: ") or DEFAULT_DATAVEC
        datavec = to_bool(datavec)

        # DL4J core usage
        DEFAULT_DL4J = 'y'
        dl4j_core = input(
            "Do you want to work with DeepLearning4J from Python? (default 'y') [y/n]: ") or DEFAULT_DL4J
        dl4j_core = to_bool(dl4j_core)

        # Spark
        spark = input(
            "Do you need Spark for distributed computation in your application? (default 'y') [y/n]: ") or DEFAULT_SPARK
        spark = to_bool(spark)
        spark_version = DEFAULT_SPARK_MAJOR
        scala_version = DEFAULT_SCALA_VERSION
        if spark:
            spark_details = input("We use Spark {} and Scala {} by default, is this OK for you? (default 'y') [y/n]: ".format(DEFAULT_SPARK_MAJOR,
                                                                                                                              DEFAULT_SCALA_VERSION)) or DEFAULT_SPARK_DETAILS
            if not spark_details[0] in ["Y", "y"]:
                spark_version = input("Which which major Spark release would you like to use? (default '%s'): " %
                                      DEFAULT_SPARK_MAJOR) or DEFAULT_SPARK_MAJOR
                scala_version = input("Which Scala version would you like to use? (default '%s'): " %
                                      DEFAULT_SCALA_VERSION) or DEFAULT_SCALA_VERSION

        cli_out = {
            'dl4j_version': dl4j_version,
            'nd4j_backend': backend,
            'dl4j_core': dl4j_core,
            'datavec': datavec,
            'spark': spark,
            'spark_version': spark_version,
            'scala_version': scala_version
        }

        validate_config(cli_out)
        formatted_json = json.dumps(cli_out, sort_keys=False, indent=2)

        click.echo("\nThis is your current settings file " +
                   click.style("config.json", bold=True) + ":\n")
        click.echo(click.style(formatted_json, fg="green", bold=True))

        confirm = input(
            "\nDoes this look good? (default 'y') [y/n]: ") or 'yes'
        if not to_bool(confirm):
            click.echo(
                "" + click.style("Please initialize pydl4j once again", fg="red", bold=True))
            return

        set_config(cli_out)

    def install(self):
        if is_docker_available():
            use_docker = input(
                "Docker available on your system. Would you like to use docker for installation> (default 'y')[y/n]: ") or 'yes'
            if to_bool(use_docker):
                click.echo(click.style(
                    "Docker is running, starting installation.", fg="green", bold=True))
                click.echo(click.style("========\n\nNote that this might take some time to complete.\n" +
                                       "We will first pull a docker container with Maven, then install all dependencies selected with 'pydl4j init'.\n" +
                                       "After completion you can start using DL4J from Python.\n\n========", fg="green", bold=False))
                _maven_build(use_docker=True)
            else:
                click.echo(click.style("========\n\nNote that this might take some time to complete.\n" +
                                       "After completion you can start using DL4J from Python.\n\n========", fg="green", bold=False))

                _maven_build(use_docker=False)
        else:
            click.echo(
                "" + click.style("Could not detect docker on your system.", fg="red", bold=True))
            click.echo(click.style("========\n\nNote that this might take some time to complete.\n" +
                                   "After completion you can start using DL4J from Python.\n\n========", fg="green", bold=False))

            _maven_build(use_docker=False)


def handle():
    try:
        cli = CLI()
        sys.exit(cli.command_dispatcher())
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        click.echo(click.style("Error: ", fg='red', bold=True))
        traceback.print_exc()
        sys.exit()


if __name__ == '__main__':
    handle()
