import logging
from logging.config import dictConfig

LOGGING_CONFIG = dict(
  version=1,
  formatters={
    'console': {'format':
                  '[%(levelname)s] %(message)s'}
  },
  handlers={
    'console': {
      'class': 'logging.StreamHandler',
      'level': logging.DEBUG,
      'formatter': 'console',
    },
  },
  root={
    'handlers': ['console'],
    'level': logging.DEBUG,
  },
  disable_existing_loggers=False
)

dictConfig(LOGGING_CONFIG)
