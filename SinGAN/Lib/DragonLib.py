import os
from pathlib import Path


def make_parent_dir_if_not_exits(target_path):
    """
    * 파일의 상위 디렉터리가 존재하지 않는 경우 상위 경로 생성
    :param target_path: 파일명 포함 경로
    :return: None
    """

    # 경로 존재하지 않을 경우 생성
    if not os.path.exists(os.path.dirname(target_path)):
        # 경로 생성
        os.makedirs(os.path.dirname(target_path))


def get_bottom_folder(path):
    """
    * 해당 경로의 가장 하위에 있는 폴더의 이름을 반환
    :param path: 경로
    :return: 가장 하위에 있는 폴더의 이름
    """

    return Path(path).parts[-1]


def get_second_bottom_folder(path):
    """
    * 해당 경로의 하위에서 두 번째에 있는 폴더의 이름을 반환
    :param path: 경로
    :return: 하위에서 두 번째에 있는 폴더의 이름
    """

    return Path(path).parts[-2]
