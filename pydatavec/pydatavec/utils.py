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


import os
import requests
import sys
import time
import math


def _mean(x):
    s = float(sum(x))
    s /= len(x)
    return s


class ProgressBar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        if target is None:
            target = -1
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def set_value(self, current, values=None, force=False):
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            if self.target is not -1:
                numdigits = int(math.floor(math.log(self.target, 10))) + 1
                barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
                bar = barstr % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
                sys.stdout.write(bar)
                self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            perc = float(current) * 100 / self.target
            info = ''
            if current < self.target and self.target is not -1:
                info += ' - %f%%' % perc
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = _mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = _mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def update(self, n=1, values=None):
        self.set_value(self.seen_so_far + n, values)


def download_file(url, file_name):
    #u = urlopen(url)
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
            file_exists = True
        else:
            print("File corrupt. Downloading again.")
            os.remove(file_name)
    if not file_exists:
        factor = int(math.floor(math.log(file_size)/math.log(1024)))
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
            #status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            #status = status + chr(8)*(len(status)+1)
            # print(status)
        f.close()
    else:
        print("File already exists - " + file_name)
        return True
