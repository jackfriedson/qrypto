import logging
import time
from typing import Tuple

import pandas as pd

from qrypto.types import Timestamp


log = logging.getLogger(__name__)


def to_unixtime(ts: Timestamp) -> int:
    if isinstance(ts, pd.Timestamp):
        ts = ts.astype(int)
    return ts


def read_key_from(path: str) -> Tuple[str, str]:
    with open(path, 'rb') as f:
        key = f.readline().strip()
        secret = f.readline().strip()
    return key, secret


class retry_on_status_code(object):
    def __init__(self, status_codes, max_retries=5, backoff_factor=0.5):
        if type(status_codes) is int:
            status_codes = [status_codes]

        self.status_codes = status_codes
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def should_retry(self, resp):
        return resp.status_code in self.status_codes

    def __call__(self, func):
        def with_retries(*args, **kwargs):
            retries = 0
            backoff = self.backoff_factor

            while True:
                resp = func(*args, **kwargs)
                if not self.should_retry(resp) or retries >= self.max_retries:
                    break

                retries += 1
                log.debug('Received status code %d -- Retrying in %f seconds...',
                          resp.status_code, backoff)
                time.sleep(backoff)
                backoff *= 2

            return resp
        return with_retries


class retry_on_exception(object):
    def __init__(self, exc_classes, max_retries=3, backoff_factor=0.5):
        if type(exc_classes) is type:
            exc_classes = [exc_classes]

        self.exc_classes = tuple(exc_classes)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def __call__(self, func):
        def with_retries(*args, **kwargs):
            retries = 0
            backoff = self.backoff_factor

            while True:
                try:
                    resp = func(*args, **kwargs)
                except self.exc_classes as e:
                    if retries >= self.max_retries:
                        raise
                    retries += 1
                    log.debug('Caught %s -- Retrying in %f seconds...',
                              e.__class__.__name__, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    break
            return resp
        return with_retries
