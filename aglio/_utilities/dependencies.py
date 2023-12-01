class DependencyChecker:
    _has_cartopy = None

    @property
    def has_cartopy(self):
        if self._has_cartopy is None:
            try:
                import cartopy  # noqa: F401

                self._has_cartopy = True
            except ImportError:
                self._has_cartopy = False
        return self._has_cartopy

    _has_yt = None

    @property
    def has_yt(self):
        if self._has_yt is None:
            try:
                import yt  # noqa: F401

                self._has_yt = True
            except ImportError:
                self._has_yt = False
        return self._has_yt

    _has_requests = None

    @property
    def has_requests(self):
        if self._has_requests is None:
            try:
                import requests  # noqa: F401

                self._has_requests = True
            except ImportError:
                self._has_requests = False
        return self._has_requests

    _has_pooch = None

    @property
    def has_pooch(self):
        if self._has_pooch is None:
            try:
                import pooch  # noqa: F401

                self._has_pooch = True
            except ImportError:
                self._has_pooch = False
        return self._has_pooch

    def requires(self, module_name, func):
        def wrapper(*args, **kwargs):
            att_name = f"has_{module_name}"
            if getattr(self, att_name, None):
                return func(*args, **kwargs)
            else:
                raise ImportError(f"This method requires {module_name}.")

        return wrapper


dependency_checker = DependencyChecker()
