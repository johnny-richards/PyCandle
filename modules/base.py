class ModuleBase(object):
    def __init__(self, *args, **kwargs):
        self.params = {}
        self.grads = {}
        self.results = {}
    
    def forward(self, *args, **kwargs):
        pass
    
    def backward(self, *args, **kwargs):
        pass