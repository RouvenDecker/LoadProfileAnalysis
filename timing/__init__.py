import time


class timer:
    '''
    basic timer class for performance measure,
    timer starts on creation
    '''

    def __init__(self) -> None:
        self._start : float = time.perf_counter()
        self._end : float = 0
        self._duration : float = 0

    def stop(self) -> None:
        '''
        stop the timing
        '''
        self._end = time.perf_counter()

    def duration(self) -> float:
        '''
        get the duration in seconds

        Returns
        -------
        float
            durationtime in seconds
        '''
        return round(self._end - self._start, 2)
