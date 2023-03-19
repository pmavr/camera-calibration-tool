import cv2
import pickle
from pathlib import Path

def show_image(img_list, msg_list=None):
    """
    Display N images. Esc char to close window. For debugging purposes.
    :param img_list: A list with images to be displayed.
    :param msg_list: A list with title for each image to be displayed. If not None, it has to be of equal length to
    the image list.
    :return:
    """
    if not isinstance(img_list, list):
        return 'Input is not a list.'

    if msg_list is None:
        msg_list = [f'{i}' for i in range(len(img_list))]
    else:
        msg_list = [f'{msg}' for msg in msg_list]

    for i in range(len(img_list)):
        cv2.imshow(msg_list[i], img_list[i])

    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    for msg in msg_list:
        cv2.destroyWindow(msg)

def get_project_root():
    '''
    :return:  path without slash in the end.
    '''
    path = f'{Path(__file__).parent}/'
    return path


def get_court_template():
    '''Using soccernet line naming convention.'''
    filename = f'{get_project_root()}court_template.pkl'
    file = open(filename, 'rb')
    file_content = pickle.load(file)
    file.close()
    return file_content