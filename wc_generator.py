#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import argparse
import os
import re
import math
import json
from datetime import *

import matplotlib.pyplot as plt
from matplotlib.lines       import Line2D
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

def plot_message_count(dataframe, time_frame = '1D', trendline = False, write_path = None):
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
    
def get_general_stats(df, print_stats = False):
    """ Gets and prints general statistics from the df:
    - Total messages sent per user
    - Total words sent per user
    - Most active day(most messages sent) per user
    - Most active week(most messages sent) per user
    - Average messages per day per user
    - Average messages per week per user
    - Average message length(chars) per user
    - Average message length(words) per user
    Returned in a dict.

    Parameters:
    ----------
    df:  Required. Dataframe containing chat information. 
    print_stats: Bool, whether or not to print the statistics.
    """
    stat_dict = {}
    for sender in df.sender.unique():
        stat_dict[sender] = {}

    # Total messages sent
    for sender in df.sender.unique():
        stat_dict[sender]['Total_sent'] = len(df.loc[df.sender == sender])
    
    # Total words sent
    for sender, group in df.groupby('sender'):
        message_lengths = group['raw_text'].map(lambda x: len(re.findall(r'\w+', x)) )
        stat_dict[sender]['Total_words_sent'] = math.ceil(message_lengths.sum())

    # Most active day
    for sender, group in df.groupby('sender'):
            # Set index to timestamp, resample per day, get message count per day
            group = group.set_index('timestamp')
            group = group.resample('1D').count().reset_index()
            # Select row of maximum value of raw_text(now the count of messages per day)
            max_row = group.loc[group.raw_text.idxmax()]
            # Select date from row
            max_timestamp = max_row.timestamp.date()
            stat_dict[sender]['Most_active_day'] = max_timestamp

    # Most active week
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('7D').count().reset_index()
        max_row = group.loc[group.raw_text.idxmax()]
        max_timestamp = max_row.timestamp.date()
        stat_dict[sender]['Most_active_week'] = max_timestamp

    # Average messages per day
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('1D').count().reset_index()
        average_sent = group.raw_text.mean()
        stat_dict[sender]['Mean_daily_sent'] = math.ceil(average_sent)
    
    # Average messages per week
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('7D').count().reset_index()
        average_sent = group.raw_text.mean()
        stat_dict[sender]['Mean_weekly_sent'] = math.ceil(average_sent)

    # Average message length(chars)
    for sender, group in df.groupby('sender'):
        message_lengths = group['raw_text'].map(lambda x: len(x))
        stat_dict[sender]['Mean_msg_len_char'] = math.ceil(message_lengths.mean())

    # Average message length(words)
    for sender, group in df.groupby('sender'):
        message_lengths = group['raw_text'].map(lambda x: len(re.findall(r'\w+', x)) )
        stat_dict[sender]['Mean_msg_len_word'] = math.ceil(message_lengths.mean())

    if print_stats:
        for sender in stat_dict:
            stat_dict[sender]['Most_active_day'] = stat_dict[sender]['Most_active_day'].strftime('%a %d, %b %Y')
            start_date = stat_dict[sender]['Most_active_week']
            end_date = start_date + timedelta(days = 7)
            start_date = start_date.strftime('%a %d, %b %Y')
            end_date = end_date.strftime('%a %d, %b %Y')
            stat_dict[sender]['Most_active_week'] = start_date + " until " + end_date
        print(json.dumps(stat_dict, indent = 2))
    
    return stat_dict

def plot_msg_len_distrib(df):
    pass

def plot_active_hour(df):
    pass

def analyse_sentiment(df):
    nltk.download('vader_lexicon')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    neutral, negative, positive = 0, 0, 0

    num_charts = len(df.sender.unique())
    fig, axs = plt.subplots(1, num_charts)
    fig.set_size_inches(11, 5)
    chart_index = 0

    for sender, group in df.groupby('sender'):
        all_text = group.raw_text
        for index, sentence in enumerate(all_text):
            if index % 10 == 0:
                print("Processing {0}%".format(int((index * 100) / len(all_text))))

            if re.match(r'^[\w]', sentence) is None:
                continue

            scores = sentiment_analyzer.polarity_scores(sentence)
            scores.pop('compound', None)

            maxAttribute = max(scores, key=lambda k: scores[k])

            if maxAttribute == "neu":
                neutral += 1
            elif maxAttribute == "neg":
                negative += 1
            else:
                positive += 1

        total = neutral + negative + positive
        print("Negative: {0}% | Neutral: {1}% | Positive: {2}%".format(
            int(negative*100/total), int(neutral*100/total), int(positive*100/total)))

        labels = ['Neutral', 'Negative', 'Positive']
        sizes = [neutral, negative, positive]
        colors = ['#00bcd7', '#F57C00', '#CDDC39']
        explode = (0.3, 0.3, 0.3)

        # Plot
        if chart_index % 2 == 0:
            wedges, texts = axs[chart_index].pie(sizes, wedgeprops=dict(width=0.5),  
                                 shadow = True, colors=colors, counterclock = False, startangle=120)
        else:
            wedges, texts = axs[chart_index].pie(sizes, wedgeprops=dict(width=0.5),
                                 shadow = True, colors=colors, counterclock = True, startangle=60)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                  bbox=bbox_props, zorder=0, va="center")

        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            axs[chart_index].annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)

        axs[chart_index].set_axis_off()
        axs[chart_index].set_title(sender)
        chart_index += 1

    fig.suptitle('Do you text happy?')
    fig.tight_layout()
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
    plot_message_count(df, '1D', trendline = True)
    get_general_stats(df, print_stats = True)
    analyse_sentiment(df)
