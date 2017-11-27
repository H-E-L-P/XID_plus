# -*- coding: utf-8 -*-

import subprocess
from os.path import dirname

import pickle


def git_version():
    """Returns the git version of the module
    This function returns a string composed of the abbreviated Git hash of the
    module source, followed by the date of the last commit.  If the source has
    some local modifications, “ [with local modifications]” is added to the
    string.
    This is used to print the exact version of the source code that was used
    inside a Jupiter notebook.
    """
    module_dir = dirname(__file__)

    command_hash = "cd {} && git rev-list --max-count=1 " \
        "--abbrev-commit HEAD".format(module_dir)
    command_date = "cd {} && git log -1 --format=%cd" \
        .format(module_dir)
    command_modif = "cd {} && git diff-index --name-only HEAD" \
        .format(module_dir)

    try:
        commit_hash = subprocess.check_output(command_hash, shell=True)\
            .decode('ascii').strip()
        commit_date = subprocess.check_output(command_date, shell=True)\
            .decode('ascii').strip()
        commit_modif = subprocess.check_output(command_modif, shell=True)\
            .decode('ascii').strip()

        version = "{} ({})".format(commit_hash, commit_date)
        if commit_modif:
            version += " [with local modifications]"
    except subprocess.CalledProcessError:
        version = "Unable to determine version."

    return version



def save(priors, posterior, filename):
    """
    Save xidplus priors and posterior data
    
    :param priors: list of prior classes
    :param posterior: posterior class
    :param filename: filename to save to
    """
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump({'priors':priors, 'posterior': posterior, 'version':git_version()}, f)

def load(filename):

    """

    :param filename: filename of xidplus data to load
    :return: list of prior classes and posterior class
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
        priors=obj['priors']
        posterior=obj['posterior']

    return priors,posterior

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))
