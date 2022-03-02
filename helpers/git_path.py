import os.path as osp


def get_git_path() -> osp:
    """
    a get function that returns the path to the git dir
    :return: git_dir: os.path
    """
    current_dir = osp.dirname(osp.realpath(__file__))
    git_dir = osp.dirname(current_dir)
    return git_dir
