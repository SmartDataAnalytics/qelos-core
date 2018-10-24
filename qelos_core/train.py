import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset, TensorDataset as PytorchTensorDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import qelos_core as q
from qelos_core.util import isnumber, isstring, ticktock, issequence


def recmap(x, mapf):      # datastructure, mapping function for elements
    if isinstance(x, dict):
        for k in x:
            x[k] = recmap(x[k], mapf)
        return x
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = recmap(x[i], mapf)
        return x
    elif isinstance(x, tuple):
        newtup = []
        for i in range(len(x)):
            newtup.append(recmap(x[i], mapf))
        newtup = tuple(newtup)
        return newtup
    elif isinstance(x, set):
        newset = set()
        for k in x:
            newset.add(recmap(k, mapf))
        return newset
    else:
        return mapf(x)


class TensorDataset(PytorchTensorDataset):
    def __init__(self, *x):
        """
        :param x: tensors in torch or numpy (converted to tensors). Last tensor must be gold.
        """
        print("WARNING! don't use this")
        tensors = []
        for xe in x:
            if isinstance(xe, np.ndarray):
                xe = torch.tensor(xe)
            tensors.append(xe)
        for xe in tensors:
            assert(xe.size(0) == tensors[0].size(0))
        super(TensorDataset, self).__init__(*tensors)


class EventEmitter(object):

    def __init__(self, **kw):
        super(EventEmitter, self).__init__(**kw)
        # events
        self._event_callbacks = {}

    def hook(self, f, *es, **kw):
        """ f to be called when e happens. Returns deleter for bound f
            can also pass pytorch's lr schedulers
        """
        if isinstance(f, AutoHooker):
            if len(es) > 0:
                raise q.SumTingWongException("can't hook autohooker explicitly on hooks")
            hookdic = f.get_hooks(self)
        else:
            hookdic = dict(zip(es, [f]*len(es)))

        for e, fe in hookdic.items():
            if e not in self._event_callbacks:
                self._event_callbacks[e] = []
            self._event_callbacks[e].append(fe)
        def deleter():
            for e, fe in hookdic.items():
                self._event_callbacks[e].remove(fe)
        # TODO: implement unhooking mechanism
        return self

    def do_callbacks(self, e):
        if not e in self._event_callbacks:
            return
        for f in self._event_callbacks[e]:
            f(self)


class TrainEventEmitter(EventEmitter):
    def hook(self, f, *es, **kw):
        """
        Same as EventEmitter with some sugar for LR schedulers:
        if "f" is a torch _LRScheduler (superclass of all schedulers),
            wraps the scheduler in a special autohooker that executes at the beginning of ever epoch.
        if "f" is a torch ReduceLROnPlateau, expects a function or LossWrapper as second argument,
            wraps both in an autohooker that at the end of every epoch evaluates provided function
            or average loss over whole epoch of the LossWrapper and lets the ReduceLROnPlateau to reduce LR.
        """
        # special hooker wrappers
        if isinstance(f, torch.optim.lr_scheduler.ReduceLROnPlateau):
            assert (len(es) == 1)
            return super(TrainEventEmitter, self).hook(_ReduceLROnPlateauAutoHooker(f, es[0]))
        elif isinstance(f, torch.optim.lr_scheduler._LRScheduler):
            return super(TrainEventEmitter, self).hook(_LRSchedulerAutoHooker(f, **kw))
        # normal hooking
        else:
            return super(TrainEventEmitter, self).hook(f, *es, **kw)


def no_losses(n):
    return tuple([q.SelectedLinearLoss(i) for i in range(n)])


class LossWrapper(EventEmitter):
    """ Wraps a normal loss with aggregating and other functionality
        TODO: refactor to work with this only, instead of lossarray """
    BEFORE_PUSH = 1
    AFTER_PUSH = 2

    def __init__(self, loss, name=None, mode="mean", **kw):
        """
        :param loss:    actual loss class
        :param name:    name for this loss (class name by default)
        :param mode:    "mean" or "sum"
        :param kw:
        """
        super(LossWrapper, self).__init__()
        self.loss, self.aggmode = loss, mode
        self.name = name if name is not None else loss.__class__.__name__

        self.agg_history = []
        self.agg_epochs = []

        self.epoch_agg_values = []
        self.epoch_agg_sizes = []

    def get_epoch_error(self):
        """ returns the aggregated error for this epoch so far """
        if self.aggmode == "mean":
            if len(self.epoch_agg_sizes) == 0:
                ret = 0.
            else:
                total = sum(self.epoch_agg_sizes)
                fractions = [x/total for x in self.epoch_agg_sizes]
                parts = [x * y for x, y in zip(self.epoch_agg_values, fractions)]
                ret = sum(parts)
        else:
            ret = sum(self.epoch_agg_values)
        ret = self.post_agg_epoch(ret)
        return ret

    def push_epoch_to_history(self, epoch=None):
        self.agg_history.append(self.get_epoch_error())
        if epoch is not None:
            self.agg_epochs.append(epoch)

    def __call__(self, pred, gold, **kw):
        l = self.loss(pred, gold, **kw)

        numex = pred.size(0) if not q.issequence(pred) else pred[0].size(0)
        if isinstance(l, tuple) and len(l) == 2:     # loss returns numex too
            numex = l[1]
            l = l[0]
        if isinstance(l, torch.Tensor):
            lp = l.item()
        else:
            lp = l
        self.epoch_agg_values.append(lp)
        self.epoch_agg_sizes.append(numex)
        return l

    def post_agg_epoch(self, x):
        if hasattr(self.loss, "post_agg_epoch"):
            x = self.loss.post_agg_epoch(x)
        return x

    def device(self, device):
        self.loss.to(device)

    def _reset(self):   # full reset
        self.reset_agg()
        self.agg_history = []
        self.agg_epochs = []

    def reset_agg(self):    # reset epoch stats
        self.epoch_agg_values = []
        self.epoch_agg_sizes = []


class eval(object):
    """ to get model predictions in a batched manner """
    def __init__(self, model):
        super(eval, self).__init__()
        self.model = model
        self._device = torch.device("cpu")
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.dataloader = None
        self.tt = ticktock("eval")

    def device(self, device):
        """ device for created data"""
        self._device = device
        return self

    def initialize(self):
        print("WARNING: setting model's device, instance might change !")
        self.model.to(self._device)

    def on(self, dataloader):
        self.dataloader = dataloader
        return self

    def set_batch_transformer(self, input_transform=None, output_transform=None):
        self.transform_batch_inp = input_transform
        self.transform_batch_out = output_transform
        return self

    def reset(self):
        return self

    def run(self):
        self.reset()
        self.initialize()
        ret = self.evalloop()
        return ret

    def evalloop(self):
        self.tt.tick("testing")
        tt = ticktock("-")
        totaltestbats = len(self.dataloader)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                batch = (batch,) if not q.issequence(batch) else batch
                batch = recmap(batch, lambda x: x.to(self._device) if isinstance(x, torch.Tensor) else x)
                # batch = [batch_e.to(self._device) for batch_e in batch]
                if self.transform_batch_inp is not None:
                    batch = self.transform_batch_inp(*batch)

                batch_reset(self.model)
                modelouts = self.model(*batch)

                if self.transform_batch_out is not None:
                    modelouts = self.transform_batch_out(modelouts)

                tt.live("eval - [{}/{}]"
                    .format(
                    i + 1,
                    totaltestbats
                )
                )
                outs.append(modelouts)
        ttmsg = "eval done"
        tt.stoplive()
        tt.tock(ttmsg)
        self.tt.tock("tested")
        out = torch.cat(outs, 0)
        return out


class AutoHooker(object):
    def get_hooks(self, emitter):
        raise NotImplemented()


class LoopRunner(object):
    """ Abstract class for different loop runners.
        Loop runners run loops like trainer and tester. """
    pass


# region basic training loop
class BasicRunner(LoopRunner, TrainEventEmitter):
    START = 0
    END = 1
    START_EPOCH = 2
    END_EPOCH = 3
    START_TRAIN = 4
    END_TRAIN = 5
    START_VALID = 6
    END_VALID = 7
    """ Takes a single trainer and an optional single validator.
        Runs them such that epochs are interleaved.
        Prints epoch lines """      # TODO: support multiple validators
    def __init__(self, trainer, validator=None):
        """ validator can be a q.tester() or any function (doesn't take arguments) """
        super(BasicRunner, self).__init__()
        self.trainer = trainer
        self.validator = validator
        self._logger = None

    @property
    def current_epoch(self):
        return self.trainer.current_epoch

    def log(self, logger):
        self._logger = logger
        return self

    def run(self, epochs=None, validinter=1, print_on_valid_only=False):
        self.trainer.pre_run()
        if isinstance(self.validator, tester):
            self.validator.pre_run()
        if epochs is not None:
            self.trainer.epochs(epochs)
        self.runloop(validinter=validinter, print_on_valid_only=print_on_valid_only)
        self.trainer.post_run()
        if isinstance(self.validator, tester):
            self.validator.post_run()
        if self._logger is not None:
            self._logger.liner_close("train.txt")

    def runloop(self, validinter=1, print_on_valid_only=False):
        tt = q.ticktock("runner")
        self.do_callbacks(self.START)
        validinter_count = 0
        while self.trainer.stop_training is not True:
            tt.tick()
            self.do_callbacks(self.START_EPOCH)
            self.do_callbacks(self.START_TRAIN)
            self.trainer.do_epoch()
            ttmsg = "Epoch {}/{} -- train: {}" \
                .format(
                self.trainer.current_epoch,
                self.trainer.max_epochs,
                pp_epoch_losses(*self.trainer.losses)
            )
            self.do_callbacks(self.END_TRAIN)
            validepoch = False
            if self.validator is not None and validinter_count % validinter == 0:
                self.do_callbacks(self.START_VALID)
                if isinstance(self.validator, tester):
                    self.validator.do_epoch(self.trainer.current_epoch, self.trainer.max_epochs)
                    ttmsg += " -- {}" \
                        .format(pp_epoch_losses(*self.validator.losses))
                else:
                    toprint = self.validator()
                    ttmsg += " -- {}".format(toprint)
                self.do_callbacks(self.END_VALID)
                validepoch = True
            self.do_callbacks(self.END_EPOCH)
            validinter_count += 1
            if not print_on_valid_only or validepoch:
                tt.tock(ttmsg)
                if self._logger is not None:
                    self._logger.liner_write("losses.txt", ttmsg)
        self.do_callbacks(self.END)


def train(trainer, validator=None):
    return BasicRunner(trainer, validator=validator)
# endregion


def batch_reset(module):        # performs all resetting operations on module before using it in the next batch
    q.rec_reset(module)
    for modu in module.modules():
        if hasattr(modu, "batch_reset"):
            modu.batch_reset()


def pp_epoch_losses(*losses:LossWrapper):
    values = [loss.get_epoch_error() for loss in losses]
    ret = " :: ".join("{:.4f}".format(value) for value in values)
    return ret


class Modelholder(TrainEventEmitter, AutoHooker):
    def __init__(self, model, **kw):
        super(Modelholder, self).__init__(**kw)
        self.model = model
        self.losses = None
        self._device = torch.device("cpu")
        self.transform_batch_inp = None
        self.transform_batch_out = None
        self.transform_batch_gold = None
        self.dataloader = None

    def device(self, device):
        self._device = device
        return self

    def initialize(self):
        print("WARNING: setting device of model and loss ! (beware: instance might change)")
        for loss in self.losses:
            loss.device(self._device)
        self.model.to(self._device)
        self.do_callbacks(self.INIT)

    def on(self, dataloader):
        self.dataloader = dataloader
        return self

    def loss(self, *args):      # TODO: supports normal PyTorch losses too ???
        # can be unspecified
        if len(args) == 1 and isinstance(args[0], int):
            args = q.no_losses(args[0])

        self.losses = [LossWrapper(loss) if not isinstance(loss, LossWrapper) else loss for loss in args]
        return self

    @property
    def no_gold(self):
        all_linear = True
        some_linear = False
        for loss in self.losses:
            if isinstance(loss.loss, (q.LinearLoss, q.SelectedLinearLoss)):
                some_linear = True
            else:
                all_linear = False
        assert(all_linear == some_linear)
        return all_linear

    def set_batch_transformer(self, input_transform=None, output_transform=None, gold_transform=None):
        self.transform_batch_inp = input_transform
        self.transform_batch_out = output_transform
        self.transform_batch_gold = gold_transform
        return self

    def reset(self):
        for loss in self.losses:
            loss._reset()
        self.do_callbacks(self.RESET)
        return self

    def pre_run(self):
        self.reset()
        self.initialize()
        self.do_callbacks(self.START)
        for loss in self.losses:
            loss._reset()

    def post_run(self):
        self.do_callbacks(self.END)


class trainer(Modelholder):
    """
    Trainer.
    Holds a model, lossarray, dataloader, batchtransformers.
    Supports compatible AutoHookers.
    """
    START = 0
    END = 1
    START_EPOCH = 2
    END_EPOCH = 3
    START_TRAIN = 4
    END_TRAIN = 5
    START_BATCH = 8
    END_BATCH = 9
    RESET = 15
    INIT = 16
    BEFORE_OPTIM_STEP = 12
    AFTER_OPTIM_STEP = 13

    def __init__(self, model, **kw):
        super(trainer, self).__init__(model, **kw)
        self.max_epochs = None
        self.current_epoch = 0
        self.stop_training = None
        self.tt = ticktock("trainer")

    def optimizer(self, optimizer):
        self.optim = optimizer
        return self

    def epochs(self, epochs):
        self.max_epochs = epochs
        return self

    # region LOOPS
    def inf_batches(self, with_info=True):
        """
        iteration over this produces infinite batches from this trainer's dataloader
        returns <batch_data>, (<batch_number>, <epoch_number>) if with_info=True
            else just <batch_data>
        """
        epoch = 0
        while True:
            for i, _batch in enumerate(self.dataloader):
                if with_info:
                    yield _batch, (i, epoch)
                else:
                    yield _batch
            epoch += 1

    def do_batch(self, _batch, i=-1):
        """
        performs a single batch of SGD on the provided batch
        with configured model, dataloader and optimizer
        """
        self.do_callbacks(self.START_BATCH)
        self.optim.zero_grad()
        self.model.train()
        # params = q.params_of(self.model)

        _batch = (_batch,) if not q.issequence(_batch) else _batch
        _batch = recmap(_batch, lambda x: x.to(self._device) if isinstance(x, torch.Tensor) else x)
        # _batch = [batch_e.to(self._device) for batch_e in _batch]
        if self.transform_batch_inp is not None:
            batch = self.transform_batch_inp(*_batch)
        else:
            batch = _batch

        if self.no_gold:
            batch_in = batch
            gold = None
        else:
            batch_in = batch[:-1]
            gold = batch[-1]

        batch_reset(self.model)
        modelouts = self.model(*batch_in)

        modelout2loss = modelouts
        if self.transform_batch_out is not None:
            modelout2loss = self.transform_batch_out(modelouts)

        if self.transform_batch_gold is not None:
            gold = self.transform_batch_gold(gold)

        trainlosses = [loss_obj(modelout2loss, gold) for loss_obj in self.losses]

        # TODO: put in penalty mechanism

        cost = trainlosses[0]

        if torch.isnan(cost).any():
            print("Cost is NaN!")
            q.embed()

        penalties = 0.
        penalty_values = {}
        for penalty_module in q.gather_penalties(self.model):
            pen = penalty_module.get_penalty()
            penalties = penalties + pen
            if type(penalty_module) not in penalty_values:
                penalty_values[type(penalty_module)] = 0.
            penalty_values[type(penalty_module)] += pen.detach().item()

        cost = cost + penalties

        cost.backward()

        self.do_callbacks(self.BEFORE_OPTIM_STEP)
        self.optim.step()
        self.do_callbacks(self.AFTER_OPTIM_STEP)

        ttmsg = "train - Epoch {}/{} - [{}/{}]: {}".format(
                    self.current_epoch+1,
                    self.max_epochs,
                    i+1,
                    len(self.dataloader),
                    pp_epoch_losses(*self.losses),
                    )
        if len(penalty_values) > 0:
            ttmsg += " " + " ".join(["+{}={:.4f}".format(k.__pp_name__, v) for k, v in penalty_values.items()])
        self.do_callbacks(self.END_BATCH)
        return ttmsg

    def do_epoch(self, tt=q.ticktock("-")):
        self.stop_training = self.current_epoch + 1 == self.max_epochs
        for loss in self.losses:
            loss.push_epoch_to_history(epoch=self.current_epoch-1)
            loss.reset_agg()
        # tt.tick()
        self.do_callbacks(self.START_EPOCH)
        self.do_callbacks(self.START_TRAIN)

        for i, _batch in enumerate(self.dataloader):
            ttmsg = self.do_batch(_batch, i=i)
            tt.live(ttmsg)

        tt.stoplive()
        self.do_callbacks(self.END_TRAIN)
        ttmsg = "Epoch {}/{} -- train: {}"\
            .format(
                self.current_epoch+1,
                self.max_epochs,
                pp_epoch_losses(*self.losses)
            )
        # tt.tock(ttmsg)
        self.do_callbacks(self.END_EPOCH)
        self.current_epoch += 1
        return ttmsg

    def trainloop(self):
        if self.max_epochs == 0:
            self.tt.msg("skipping training")
            return
        self.stop_training = False
        self.tt.tick("training")
        tt = ticktock("-")
        while not self.stop_training:
            tt.tick()
            ttmsg = self.do_epoch(tt=tt)
            tt.tock(ttmsg)
        self.tt.tock("trained")
    # endregion

    def reset(self):
        self.current_epoch = 0
        super(trainer, self).reset()

    def run(self):
        self.pre_run()
        self.trainloop()
        self.post_run()

    # region AutoHooker interface  -- how it hooks into a loop runner
    def get_hooks(self, ee):
        return {ee.END_EPOCH: self.on_end_epoch,
                ee.RESET: self.on_reset,
                ee.INIT: self.on_init}

    def on_reset(self, owner, **kw):    self.reset()
    def on_init(self, owner, **kw):     self.initialize()

    def on_end_epoch(self, owner, **kw):
        if not isinstance(owner, trainer):
            raise q.SumTingWongException("can only be hooked to a trainer")
        epoch = owner.current_epoch
        maxepochs = owner.epochs
        self.do_next_iter(epoch, maxepochs)
    # endregion


class tester(Modelholder):
    START = 0
    END = 1
    START_TEST = 4
    END_TEST = 5
    START_EPOCH = START_TEST
    END_EPOCH = END_TEST
    START_BATCH = 8
    END_BATCH = 9
    RESET = 15
    INIT = 16

    _name = "tester"

    def __init__(self, model, **kw):
        super(tester, self).__init__(model, **kw)
        self.tt = ticktock(self._name)

    def run(self):
        self.pre_run()
        ret = self.testloop()
        self.post_run()
        return ret

    def do_epoch(self, epoch=None, maxepoch=None):
        epochs = None if epoch is None and maxepoch is None else (epoch, maxepoch)
        results = self.testloop(epoch=epochs)

    def testloop(self, epoch=None):
        if epoch is None:
            self.tt.tick("testing")
        tt = ticktock("-")
        self.model.eval()
        self.do_callbacks(self.START_TEST)
        for loss_obj in self.losses:
            loss_obj.push_epoch_to_history()
            loss_obj.reset_agg()
        totalbats = len(self.dataloader)
        for i, _batch in enumerate(self.dataloader):
            self.do_callbacks(self.START_BATCH)

            _batch = (_batch,) if not q.issequence(_batch) else _batch
            _batch = recmap(_batch, lambda x: x.to(self._device) if isinstance(x, torch.Tensor) else x)
            # _batch = [batch_e.to(self._device) for batch_e in _batch]
            if self.transform_batch_inp is not None:
                batch = self.transform_batch_inp(*_batch)
            else:
                batch = _batch

            if self.no_gold:
                batch_in = batch
                gold = None
            else:
                batch_in = batch[:-1]
                gold = batch[-1]

            batch_reset(self.model)
            modelouts = self.model(*batch_in)

            modelout2loss = modelouts
            if self.transform_batch_out is not None:
                modelout2loss = self.transform_batch_out(modelouts)

            if self.transform_batch_gold is not None:
                gold = self.transform_batch_gold(gold)

            losses = [loss_obj(modelout2loss, gold) for loss_obj in self.losses]

            epochmsg = ""
            if epoch is not None:
                curepoch, maxepoch = epoch
                epochmsg = "Epoch {}/{} -".format(curepoch, maxepoch)

            tt.live("{} - {}[{}/{}]: {}"
                .format(
                self._name,
                epochmsg,
                i + 1,
                totalbats,
                pp_epoch_losses(*self.losses)
            )
            )
            self.do_callbacks(self.END_BATCH)
        # losses = self.losses.get_agg_errors()
        tt.stoplive()
        ttmsg = "{}: {}" \
            .format(
            self._name,
            pp_epoch_losses(*self.losses)
        )
        self.do_callbacks(self.END_TEST)
        if epoch is None:
            tt.tock(ttmsg)
            self.tt.tock("tested")
        return ttmsg

    # region AutoHooker -- how it hooks into a trainer
    def get_hooks(self, ee):
        return {ee.END_TRAIN: self.on_end_epoch,
                ee.RESET: self.on_reset,
                ee.INIT: self.on_init}

    def on_reset(self, owner, **kw):    self.reset()
    def on_init(self, owner, **kw):     self.initialize()

    def on_end_epoch(self, owner, **kw):
        if not isinstance(owner, trainer):
            raise q.SumTingWongException("can only be hooked to a trainer")
        epoch = owner.current_epoch
        maxepochs = owner.epochs
        self.do_epoch(epoch, maxepochs)
    # endregion


# region AUTOHOOKERS
class BestSaver(AutoHooker):
    def __init__(self, criterion, model, path, higher_is_better=True, autoload=False,
                 verbose=False, **kw):
        super(BestSaver, self).__init__(**kw)
        self.criterion = criterion
        self.model = model
        self.path = path
        self.higher_better = 1. if higher_is_better else -1.
        self.best_criterion = -1.
        self.verbose = verbose
        self.callbacks = {}
        self.autoload = autoload        # automatically load on END event

    def get_hooks(self, ee):
        hooks = {ee.END_EPOCH: self.save_best_model}
        if self.autoload:
            hooks[ee.END] = self.autoload_best
        return hooks

    def save_best_model(self, _, **kw):
        # assert isinstance(trainer, train)
        current_criterion = self.criterion()
        decision_value = current_criterion - self.best_criterion    # positive if current is higher
        decision_value *= self.higher_better            # higher better --> positive is higher = better
        # remark: with this way, later can extend to specifying by how much it should improve --> TODO
        if decision_value > 0:
            if self.verbose:
                print("Validation criterion improved from {} to {}. Saving model..."\
                      .format(self.best_criterion, current_criterion))
            self.best_criterion = current_criterion
            torch.save(self.model.state_dict(), self.path)

    def autoload_best(self, _, **kw):
        if self.verbose:
            print("Reloading best weights ({})".format(self.best_criterion))
        self.model.load_state_dict(torch.load(self.path))


class _LRSchedulerAutoHooker(AutoHooker):
    def __init__(self, s, verbose=False, **kw):
        super(_LRSchedulerAutoHooker, self).__init__(**kw)
        self.s = s
        self.verbose = verbose

    def get_hooks(self, ee):
        return {ee.START_EPOCH: self.on_start_epoch}

    def on_start_epoch(self, ee, **kw):
        self.s.step(epoch=ee.current_epoch)
        if self.verbose:
            print("first group lr decayed to: {}".format(self.s.optimizer.param_groups[0]["lr"]))


class _ReduceLROnPlateauAutoHooker(AutoHooker):
    """ Don't use this, just hook torch.optim.lr_scheduler.ReduceLROnPlateau into trainer,
        and provide LossWrapper or lambda function as second argument"""
    def __init__(self, s, critf, **kw):
        super(_ReduceLROnPlateauAutoHooker, self).__init__(**kw)
        self.s = s
        if isinstance(critf, LossWrapper):
            self.critf = lambda: critf.get_epoch_error()
        else:
            self.critf = critf

    def get_hooks(self, ee):
        return {ee.END_EPOCH: self.on_end_epoch}

    def on_end_epoch(self, ee, **kw):
        current_epoch = ee.current_epoch if hasattr(ee, "current_epoch") else None
        self.s.step(self.critf(), epoch=current_epoch)


class ClipGradNorm(AutoHooker):
    def __init__(self, norm, normtype=2, **kw):
        super(ClipGradNorm, self).__init__(**kw)
        self._norm = norm
        self._normtype = normtype

    def get_hooks(self, ee):
        return {trainer.BEFORE_OPTIM_STEP: self.on_before_optim_step}

    def on_before_optim_step(self, trainer, **kw):
        if self._norm is not None:
            nn.utils.clip_grad_norm_(trainer.model.parameters(), self._norm, norm_type=self._normtype)


class TrainerChainer(AutoHooker):
    def __init__(self, trainer, **kw):
        super(TrainerChainer, self).__init__(**kw)
        self._trainer = trainer

    def get_hooks(self, ee):
        return {trainer.END_BATCH: self.on_end_batch,
                trainer.START: self.on_start}

    def on_end_batch(self, *x, **kw):
        self._trainer.do_next_iter()

    def on_start(self, *x, **kw):
        self._trainer.reset()
        self._trainer.initialize()


class EarlyStopper(AutoHooker):
    def __init__(self, select, patience=0, delta=0.,
                 minepochs=5, lessisbetter=True, custom=None, **kw):
        """ select arg must be a function that returns monitored value """
        super(EarlyStopper, self).__init__(**kw)
        self.monitorf = select
        self.patience = patience
        self.delta = delta
        self.minepochs = minepochs
        self.history = []
        self.customf = custom
        self.lessisbetter = lessisbetter

    def get_hooks(self, ee):
        return {trainer.END_EPOCH: self.on_end_epoch}

    def on_end_epoch(self, trainer, **kw):
        i = trainer.current_epoch

        monval = self.monitorf()
        if monval is None:
            return

        self.history.append(monval)

        # make early stopping decision based on history (including latest)
        stop = False
        stop |= self.customf(self.history)
        if len(self.history) >= 2 + self.patience and i > self.minepochs:
            last, check = self.history[-1], self.history[-2-self.patience]
            inbetween = self.history[-self.patience-1:-1]
            if self.lessisbetter:
                _stop = last > check + self.delta
                _cancel_stop = check > min(inbetween)
            else:
                _stop = last < check + self.delta
                _cancel_stop = check < max(inbetween)
            _stop &= not _cancel_stop
            stop |= stop
        trainer.stop_training |= stop


class HyperparamScheduler(AutoHooker):
    def __init__(self, hp, **kw):
        super(HyperparamScheduler, self).__init__(**kw)
        self._hp = hp

    def get_hooks(self, ee):
        return {trainer.START_EPOCH: self.on_start_epoch}

    def on_start_epoch(self, trainer, **kw):
        pass


class EpochHyperparamScheduler(HyperparamScheduler):
    def __init__(self, hp, f, **kw):
        """ f takes epoch number and max epochs and hp and returns new value for hp """
        super(EpochHyperparamScheduler, self).__init__()
        self._f = f

    def get_hooks(self, ee):
        return {trainer.START_EPOCH: self.on_start_epoch}

    def on_start_epoch(self, trainer, **kw):
        newval = self._f(trainer.current_epoch, maxepoch=trainer.max_epochs, hp=self._hp)
        self._hp.v = newval

# endregion
