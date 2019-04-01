class Map(dict):

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def update(self, *args, **kwargs):
        '''
        Return self.
        '''
        super(Map, self).update(*args, **kwargs)
        return self

    def apply(self, func, ignored=set()):
        for key in self:
            if key in ignored:
                continue

            if isinstance(self[key], Map):
                self[key].apply(func, ignored=ignored)
            else:
                self[key] = func(self[key])
