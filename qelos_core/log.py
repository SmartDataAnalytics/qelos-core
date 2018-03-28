import qelos_core as q
import os
import json
import warnings
import pandas as pd
import glob
import codecs


OPT_PREFIX = "zexp"
OPT_SETTINGS_NAME = "settings.json"
OPT_LOG_NAME = "log{}.lines"


def get_default_log_path(prefix):
    prefix = prefix if prefix is not None else OPT_PREFIX
    i = 0
    found = False
    while not found:
        canpath = "{}{}".format(prefix, i)
        if os.path.exists(canpath):
            i += 1
            continue
        else:
            found = True
    return canpath


def get_train_dump_path(p, logname=None):
    found = False
    if logname is not None:
        canpath = p + "/" + logname
        found = True
    i = 0
    while not found:
        canpath = p + "/" + OPT_LOG_NAME.format(i if i > 0 else "")
        if os.path.exists(canpath):
            i += 1
            continue
        else:
            found = True
    return canpath


class Logger(object):
    def __init__(self, p=None, prefix=None, **kw):
        super(Logger, self).__init__()
        assert(p is None or prefix is None)
        self.p = p if p is not None else get_default_log_path(prefix)
        if os.path.exists(self.p):
            raise q.SumTingWongException("path '{}' already exists".format(p))
        else:
            os.makedirs(self.p)
        self._current_train_file = None
        self._current_numbers = []

    def save_settings(self, **kw):
        p = self.p + "/" + OPT_SETTINGS_NAME
        with open(p, "w") as f:
            json.dump(kw, f, sort_keys=True)

    def load_settings(self):
        p = self.p + "/" + OPT_SETTINGS_NAME
        with open(p) as f:
            r = json.load(f)
        return r

    def update_settings(self, **kw):
        settings = self.load_settings()
        settings.update(kw)
        self.save_settings(**settings)

    def save_lines(self, lines, filepath, use_unicode=False):
        if not use_unicode:
            with open(self.p + "/" + filepath, "w") as f:
                for line in lines:
                    f.write("{}\n".format(line))
        else:
            with codecs.open(self.p + "/" + filepath, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write("{}\n".format(line))

    def logc(self, x, looper, logfilename):
        """ keep logging state of "x" based on "eventemitter"'s events and store in "logfilename"
            smart method --> dispatches according to type of x
        """
        raise q.SumTingWongException()

    def loglosses(self, looper, logfilename):
        sublogger = LossesWriter(looper, self.p + "/" + logfilename)
        sublogger.start()


class LossesWriter(q.AutoHooker):
    """ Keeps writing lossarray on every push"""
    def __init__(self, looper, logfilepath):
        super(LossesWriter, self).__init__()
        self.path = logfilepath
        self.looper = looper
        # self.start_logging()
        self.c = 1.

    def start(self):
        self.looper.hook(self)

    def get_hooks(self):
        return {self.looper.END_EPOCH: self.on_push,
                self.looper.START: self.on_start,
                self.looper.END: self.on_end}

    def on_start(self, looper, **kw):
        self.start_logging()

    def on_end(self, looper, **kw):
        self.stop()

    def start_logging(self, names=None, logname=None, overwrite=False, **kw):
        # make writer
        if os.path.exists(self.path):
            if not overwrite:
                raise q.SumTingWongException("file already exists")
            else:
                warnings.warn("training log file already exists. overwriting {}".format(self.path))
        self._current_file = open(self.path, "w+")
        names = ["N."]
        names += [x.get_name() for x in self.looper.losses.losses]
        line = "\t".join(names) + "\n"
        self._current_file.write(line)
        self._current_file.flush()

    def stop(self):
        self._current_file.close()

    def on_push(self, looper, **kw):
        self.flush_loss_history(looper, **kw)

    def flush_loss_history(self, looper, **kw):
        numbers = [self.c] + self.looper.losses.get_agg_errors()
        line = "\t".join(["{:f}".format(n) for n in numbers]) + "\n"
        self._current_file.write(line)
        self._current_file.flush()
        self.c += 1


class ExpLog(object):
    def __init__(self, settings, data, **kw):
        super(ExpLog, self).__init__(**kw)
        self.settings = settings
        self.data = data


class ExpLogCollection(object):
    def __init__(self, explogs, **kw):
        super(ExpLogCollection, self).__init__(**kw)
        self.explogs = explogs

    def filter(self, **kw):
        pass


