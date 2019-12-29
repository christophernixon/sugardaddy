#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import argparse
import os
import re
from datetime import *

import matplotlib.pyplot as plt
from matplotlib.lines       import Line2D
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from dateutil.parser import parse
from emoji import UNICODE_EMOJI
from PIL import Image

from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud


"""
Whatsapp line format: 
"[dd/mm/yyyy, hh/mm/ss] SENDER: MESSAGE\n"
"""

def parse_texts(raw_text_file):
    """Given a filepath to a .txt file with Whatsapp formatted lines,
    this returns a dataframe representing this file"""

    id = 0
    text_table = []
    with open(raw_text_file, 'r') as file:
        for line in file:
            id += 1
            tmp_array = []

            date = _extract_date(line)
            sender = _extract_sender(line)
            length = _extract_length(line)
            message_body = _extract_message(line)
            
            # TODO: Handle multi-line texts
            if date == "" or sender == "" or length == 0:
                continue

            tmp_array.extend((id,date,sender,length,message_body))
            # tmp_array.append(date)
            # tmp_array.append(sender)
            # tmp_array.append(length)
            # tmp_array.append(message_body)
            text_table.append(tmp_array)
            # print("Processed line {}".format(id))
    data = np.array(text_table)
    dataframe = pd.DataFrame({'text_id': data[:, 0], 'timestamp': data[:, 1],
                            'sender': data[:, 2], 'length': data[:, 3],
                            'raw_text': data[:, 4]})
    return dataframe

def _extract_date(line):
    """Given a string in Whatsapp format,
    returns a datetime object of when the message was sent."""

    pattern = re.compile("(\[[^]]{20}\])")
    match = re.search(pattern, line)
    if match:
        raw_date = match.group()[1:-1]
        date = parse(raw_date, dayfirst=True)
    else:
        # TODO: Handle multi-line texts
        date = ""
    return date

def _extract_sender(line):
    """Given a string in Whatsapp format,
    returns the sender of the message."""

    pattern = re.compile("(] [^:]*:)")
    match = re.search(pattern, line)
    if match:
        sender = match.group()[2:-1]
        if is_dancer(sender):
            sender = "Clara"
    else:
        # TODO: Handle multi-line texts
        sender = ""
    return sender

def is_dancer(string):
    """Tests whether string contains :dancer: emoji."""
    if '💃' in string:
        return True
    return False

def _extract_length(line):
    """Given a string in Whatsapp format,
    returns the length of the message body of the string"""

    pattern = re.compile("(] [^:]*: )")
    match = re.search(pattern, line)
    if match:
        sender_index = match.end()
        return len(line[sender_index:])
    else:
        # TODO: Handle multi-line texts
        return len(line)

def _extract_message(line):
    """Given a string in Whatsapp format,
    returns the message body of the string."""

    pattern = re.compile("(] [^:]*: )")
    match = re.search(pattern, line)
    if match:
        sender_index = match.end()
        return line[sender_index:].rstrip()
    else:
        # TODO: Handle multi-line texts
        return line

def generate_wordcloud(dataframe, sender=None, mask_path=None, write_path=None):
    """Generates wordcloud from raw_text field of dataframe. 
    dataframe: Required. Dataframe containing chat information.
    sender: Filter text for wordcloud to only messages sent from sender.
    mask_path: Supply filepath to image to be used as mask for wordcloud.
    write_path: Supply path for wordcloud to be saved to. Default is wordcloud/images/chat{int}.png"""

    # Produce text for wordcloud
    if not sender:
        text = " ".join(raw_text for raw_text in dataframe.raw_text)
    else:
        text = " ".join(raw_text for raw_text in df[df["sender"]==sender].raw_text)

    print ("There are {} words in the combination of all raw_texts.".format(len(text)))

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["image", "omitted", "audio", "voice", "call","will","know","now", "ll", "re", "don"])

    if mask_path:
        # Generate mask
        mask = np.array(Image.open(mask_path))
        wordcloud = WordCloud(max_words=1000, background_color="white",mode="RGBA", mask=mask,stopwords=stopwords)
        # create coloring from image
        image_colors = ImageColorGenerator(mask)
        wordcloud.generate(text)
         # Display wordcloud
        plt.figure(figsize=[10,10])
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
    else:
        wordcloud = WordCloud(max_words=2000, background_color="white",mode="RGBA",stopwords=stopwords)
        wordcloud.generate(text)
        # Display wordcloud
        plt.figure(figsize=[10,10])
        plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    # store to file
    if write_path:
        plt.savefig(write_path, format="png")
        print("Saving to {}.".format(write_path))
    else:
        i = 0
        while os.path.exists("images/chat{}.png".format(i)):
            i += 1
        print("Saving to images/chat{}.png.".format(i))
        plt.savefig("images/chat{}.png".format(i), format="png")

    plt.show()

def message_count(dataframe, time_frame = '1D', trendline = False, write_path = None):
    """ Plot message count for each user for various timeframes.
    dataframe: Required. Dataframe containing chat information.
    time_frame: Can be used to get weekly message count: '7D', or monthly count '1M'.
                Default is daily count '1D'.
    trendline: Whether or not to include a trendline. Default is False.
    write_path: Supply path for plot to be saved to. Default is 'wordcloud/images/msgs_day{int}.png'.
    """

    # Setup colors and fonts
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rc('font', family = 'Comic Sans MS')
    register_matplotlib_converters()

    valid = {'1D', '7D', '1M'}
    if time_frame not in valid:
        raise ValueError("message_count: time_frame must be one of {}.\n{} supplied.".format(valid, time_frame))
    if time_frame == '1D':
        time_frame_full = 'day'
    elif time_frame == '7D':
        time_frame_full = 'week'
    else:
        time_frame_full = 'month'

    # Set index to be timestamp for resampling
    dataframe = dataframe.set_index('timestamp')
    senders = {sender: dataframe[dataframe.sender == sender] for sender in dataframe.sender.unique()}
    
    # Resample to a week by summing
    for sender in senders:
        senders[sender] = senders[sender].resample(time_frame).count().reset_index()

    # Create figure and subplot
    fig, ax = plt.subplots()
    
    # Plot sender lines
    color_index = 0
    for i, sender in enumerate(senders):
        ax.plot(senders[sender].timestamp, senders[sender].raw_text, linewidth=1.5, color = colors[i])
        color_index = i + 1

    # calculate the trendline
    if trendline:
        x = [x for x in senders[sender].timestamp.index]
        y = senders[sender].raw_text.values
        z = np.polyfit(x, y, 10)
        p = np.poly1d(z)
        ax.plot(senders[sender].timestamp, p(x), linewidth = 2, color = colors[color_index])

    # Legend and titles
    custom_lines = [Line2D([], [], color = colors[i], lw = 4, markersize = 6) for i in range(len(colors))]
    legend_list = [sender for sender in senders.keys()]
    if trendline:
        legend_list.append('Overall trend')
    ax.legend(custom_lines, legend_list, bbox_to_anchor = (1, 1), loc = 'upper right',
              framealpha = 1.0,borderaxespad=0.1, edgecolor = 'white')
    plt.title("Number of messages per {}".format(time_frame_full), fontsize = 20)
    ax.set_ylabel('# of Messages', fontsize = 16)
    ax.set_xlabel('Time({}s)'.format(time_frame_full), fontsize = 16)

    # Set size of graph
    fig.set_size_inches(20, 10)
    
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.format_xdata = mdates.DateFormatter('%b-%d')
    ax.format_ydata = lambda x: int(x)
    
    # Create horizontal grid
    ax.grid(True, axis = 'both',linewidth = 0.5)
    
    if write_path:
        fig.savefig('images/mgs_day/messages_per_day.png', format = "PNG", dpi = 100)
    else:
        i = 0
        while os.path.exists("images/mgs_day/messages_per_day{}.png".format(i)):
            i += 1
        fig.savefig('images/mgs_day/messages_per_day{}.png'.format(i), format = "PNG", dpi = 100)
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Filepath to whatsapp text file to be processed.",
                        metavar="Whatsapp file")
    parser.add_argument("-s","--sender", help="Filter text for wordcloud to only messages sent from sender.",
                        default=None,metavar="[sender]")
    parser.add_argument("-m","--mask", help="Filepath to image used for a mask for wordcloud.",
                        default=None,metavar="[mask source]")
    parser.add_argument("-d","--dest", help="Filepath to where wordcloud should be saved.",
                        default=None,metavar="[saving destination]")                    
    args = parser.parse_args()

    df = parse_texts(args.filepath)
    generate_wordcloud(df,sender = args.sender, mask_path = args.mask, write_path = args.dest)
    message_count(df, '1D', trendline = False)
