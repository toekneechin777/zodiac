"""
Callback utils
"""

class Callback:
    """
    Callback base class/interface defines functions for different stages of
    the training loop
    """
    
    def __init__(self): pass
    def on_pretrain_routine_start(self): pass
    def on_pretrain_routine_end(self): pass
    def on_train_start(self): pass
    def on_train_epoch_start(self): pass
    def on_train_batch_start(self): pass
    def optimizer_step(self): pass
    def on_before_zero_grad(self): pass
    def on_train_batch_end(self): pass
    def on_train_epoch_end(self): pass
    def on_val_start(self): pass
    def on_val_batch_start(self): pass
    def on_val_image_end(self): pass
    def on_val_batch_end(self): pass
    def on_val_end(self): pass
    def on_val_end(self): pass
    def on_fit_epoch_end(self): pass # fit = train + eval
    def on_model_save(self): pass
    def on_train_end(self): pass
    def on_params_update(self): pass


class CallbackHandler():
    def __init__(self, cbs=None):
        self._cbs = cbs if cbs else []

    def set_learn(self, learn):
        self.learn = learn
        for cb in self.cbs:
            cb.learn = self.learn

    def on_train_start(self):
        self.in_train = True
        self.learn.stop = False
        res = True
        for cb in self.cbs: res = res and cb.on_train_start()
        return res

    def on_train_epoch_start(self, epoch):
        self.learn.model.train()
        self.in_train = True
        res = True
        for cb in self.cbs: res = res and cb.on_train_epoch_start(epoch)
        return res

    def on_val_start(self):
        self.learn.model.eval()
        self.in_train = False
        res = True
        for cb in self.cbs: res = res and cb.on_val_start()
        return res
    
    def on_train_epoch_end(self):
        res = True
        for cb in self.cbs: res = res and cb.on_train_epoch_end()
        return res
    
    def on_before_zero_grad(self):
        res = self.in_train
        for cb in self.cbs:
            res = res and cb.on_before_zero_grad()
        return res

    def optimizer_step(self):
        res = True
        for cb in self.cbs: res = res and cb.optimizer_step()
        return res

    def on_train_end(self):
        res = not self.in_train
        for cb in self.cbs: res = res and cb.on_train_end()
        return res