import logging

aglio_log = logging.getLogger("aglio")
aglio_log.setLevel(logging.INFO)

_formatter = logging.Formatter("%(name)s : [%(levelname)s ] %(asctime)s:  %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(_formatter)
aglio_log.addHandler(stream_handler)
