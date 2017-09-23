import logging

class CustomFilter(logging.Filter):

    def __init__(self, key, values):
        super(CustomFilter, self).__init__()
        self.key = key

        if type(values) is list:
            self.values = values
        elif type(values) is str:
            self.values = [value.strip() for value in values.split(',')]
        else:
            raise ValueError('Expected values must be a list or string')

    def filter(self, record):
        event_name = record.get(self.key) or record.get('extra', {}).get(self.key)
        return event_name in self.values


OrderCloseFilter = CustomFilter('event_name', 'order_close')
