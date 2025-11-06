import logging
FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: str = "INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=FMT, datefmt=DATEFMT)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
