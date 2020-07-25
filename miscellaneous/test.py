#from dl_bot import DLBot
#import time
#
#telegram = True
#
#if telegram:
#    #telegram_token = "1178257144:AAH5DEYxJjPb0Qm_afbGTuJZ0-oqfIMFlmY"  # replace TOKEN with your bot's token
#    telegram_token = "1099550359:AAH7A5Bmq9Qs7cZrwMJ7ES6i3-_fu9OhDZ0"  # gigilatrottrola
#    telegram_user_id = None  # replace None with your telegram user id (integer):
#    # Create a DLBot instance
#    bot = DLBot(token=telegram_token, user_id=telegram_user_id)
#    # Activate the bot
#    bot.activate_bot()
#
#while True:
#    try:
#        bot.send_message("K-Fold finished")
#    except:
#        print("no")
#
#    time.sleep(1)
#
##===========================================================

import requests
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO
import json



def Invia_Messaggio(Messaggio):
    TokenBot = '1099550359:AAH7A5Bmq9Qs7cZrwMJ7ES6i3-_fu9OhDZ0'
    IdCanale = '-1001352516993'
    URI = 'https://api.telegram.org/bot' + TokenBot + '/sendMessage?chat_id=' + IdCanale + '&parse_mode=Markdown&text=' + Messaggio

    response = requests.get(URI)

    return json.loads(response.content)['ok']


def Invia_Foto(plt, Descrizione):

    figdata = BytesIO()
    plt.savefig(figdata, format='png')

    TokenBot = '1099550359:AAH7A5Bmq9Qs7cZrwMJ7ES6i3-_fu9OhDZ0'
    IdCanale = '-1001352516993'
    URI = 'https://api.telegram.org/bot' + TokenBot + '/sendPhoto?chat_id=' + IdCanale + "&caption=" + Descrizione

    foto = {'photo': ("Foto", figdata.getvalue(), 'image/png')}
    response = requests.get(URI, files=foto)

    return json.loads(response.content)['ok']



i = 0
while i < 15:
    Invia_Messaggio("Prova {i}".format(i=i))
    i += 1

    labellist = [0, 1, 2, 3, 4, 5, 6]
    predlist  = [0, 1, 2, 3, 4, 5, 6]

    CM = confusion_matrix(labellist, predlist, labels=[0, 1, 2, 3, 4, 5, 6])

    labels_all = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    df_cm = pd.DataFrame(CM, index=[i for i in labels_all], columns=[i for i in labels_all])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    Invia_Foto(plt, "description")


    time.sleep(1)



