"""
    Calculate and plot Swarzschild black holes with a thin accretion disk
"""

try:
    from importlib.metadata import version, metadata

    __version__ = version("luminet")
    _meta = metadata("luminet")

    # Parse author-email field (format: "Name <email@example.com>")
    author_email = _meta.get("Author-email", "")
    if "<" in author_email and ">" in author_email:
        __author__ = author_email.split("<")[0].strip()
        __email__ = author_email.split("<")[1].split(">")[0].strip()
    else:
        __author__ = _meta.get("Author", "unknown")
        __email__ = author_email or "unknown"

    __license__ = _meta.get("License", "unknown")

except Exception:
    __version__ = "unknown"
    __author__ = "unknown"
    __email__ = "unknown"
    __license__ = "unknown"
