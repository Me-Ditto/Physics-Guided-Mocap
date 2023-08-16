
    mod = imp.load_dynamic("cymj", cext_so_path)
    return mod


class custom_build_ext(build_ext):
    """
    Custom build_ext to suppress the "-Wstrict-prototypes" warning.
    It arises from the fact that we're using C++. This seems to be
    the cleanest way to get rid of the extra flag.

    See http://stackoverflow.com/a/36293331