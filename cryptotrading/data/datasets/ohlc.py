from collections import OrderedDict

class OHLCDataset(OrderedDict):

    def __init__(self, *args, **kwargs):
        """
        """
        self.max_size = kwargs.pop('max_size', 10000)
        self.last_timestamp = None
        super(OHLCDataset, self).__init__()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if len(self) >= self.max_size:
            self.popitem(last=False)

    def add_all(self, incoming_data):
        for entry in incoming_data:
            self[entry.get('time')] = entry
        self.last_timestamp = list(self.keys())[-1]

    def last_price(self):
        return self[self.last_timestamp].get('close')
