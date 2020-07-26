import os
import sys
import numpy as np
import requests
import json
import datetime


def write_ply(fn, verts, colors=0):
    if colors.any():
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    else:
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            end_header
            '''
    verts = verts.reshape(-1, 3)
    if colors.any():
        out_colors = colors.copy()
        verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        if colors.any():
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        else:
            np.savetxt(f, verts, fmt='%f %f %f ')


telegram_token = "1178257144:AAH5DEYxJjPb0Qm_afbGTuJZ0-oqfIMFlmY"  # replace TOKEN with your bot's token
telegram_channel = '-1001352516993'


def send_telegram_message(message):
    """

    Args:
        message: text

    Returns: True if ok

    """
    URI = 'https://api.telegram.org/bot' + telegram_token + '/sendMessage?chat_id=' + telegram_channel + '&parse_mode=Markdown&text=' + str(datetime.datetime.now()) + "\n" + message
    response = requests.get(URI)
    return json.loads(response.content)['ok']


def send_telegram_picture(plt, description):
    """

    Args:
        plt: matplotlib.pyplot
        description: sends a figure with the confusion matrix through the telegram channel

    Returns: True if ok

    """
    figdata = BytesIO()
    plt.savefig(figdata, format='png')
    URI = 'https://api.telegram.org/bot' + telegram_token + '/sendPhoto?chat_id=' + telegram_channel + "&caption=" + str(datetime.datetime.now()) + "\n" + description
    pic = {'photo': ("Foto", figdata.getvalue(), 'image/png')}
    response = requests.get(URI, files=pic)

    return json.loads(response.content)['ok']