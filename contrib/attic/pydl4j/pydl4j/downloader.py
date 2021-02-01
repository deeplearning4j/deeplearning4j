#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   *  See the NOTICE file distributed with this work for additional
#   *  information regarding copyright ownership.
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
################################################################################

from .progressbar import ProgressBar
import requests
import math
import os
import hashlib


def download(url, file_name):
    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-length'])
    '''
    if py3:
        file_size = int(u.getheader("Content-Length")[0])
    else:
        file_size = int(u.info().getheaders("Content-Length")[0])
    '''
    file_exists = False
    if os.path.isfile(file_name):
        local_file_size = os.path.getsize(file_name)
        if local_file_size == file_size:
            sha1_file = file_name + '.sha1'
            if os.path.isfile(sha1_file):
                print('sha1 found')
                with open(sha1_file) as f:
                    expected_sha1 = f.read()
                BLOCKSIZE = 65536
                sha1 = hashlib.sha1()
                with open(file_name) as f:
                    buff = f.read(BLOCKSIZE)
                    while len(buff) > 0:
                        sha1.update(buff)
                        buff = f.read(BLOCKSIZE)
                if expected_sha1 == sha1:
                    file_exists = True
                else:
                    print("File corrupt. Downloading again.")
                    os.remove(file_name)
            else:
                file_exists = True
        else:
            print("File corrupt. Downloading again.")
            os.remove(file_name)
    if not file_exists:
        factor = int(math.floor(math.log(file_size) / math.log(1024)))
        display_file_size = str(file_size / 1024 ** factor) + \
            ['B', 'KB', 'MB', 'GB', 'TB', 'PB'][factor]
        print("Source: " + url)
        print("Destination " + file_name)
        print("Size: " + display_file_size)
        file_size_dl = 0
        block_sz = 8192
        f = open(file_name, 'wb')
        pbar = ProgressBar(file_size)
        for chunk in r.iter_content(chunk_size=block_sz):
            if not chunk:
                continue
            chunk_size = len(chunk)
            file_size_dl += chunk_size
            f.write(chunk)
            pbar.update(chunk_size)
            # status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            # status = status + chr(8)*(len(status)+1)
            # print(status)
        f.close()
    else:
        print("File already exists - " + file_name)
        return True
