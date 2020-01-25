"""Functions for creating various plots based on a whatsapp text file dataframe."""
import json
import math
import os
import re
import logging
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nltk
import numpy as np
from matplotlib.lines import Line2D
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas.plotting import register_matplotlib_converters
from PIL import Image

import helper_functions as hf
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

# Setup logging to file and console.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("logs/{}.log".format(__name__), mode='w')
c_handler.setLevel(logging.WARNING)
# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

def generate_wordcloud(df, sender=None, mask_path=None, write_path=None):
    """Generate wordcloud from raw_text field of dataframe.

    dataframe: Required. Dataframe containing chat information.
    sender: Filter text for wordcloud to only messages sent from sender.
    mask_path: Supply filepath to image to be used as mask for wordcloud.
    write_path: Supply path for wordcloud to be saved to.
                Default is wordcloud/images/chat{int}.png
    """
    # Produce text for wordcloud
    if not sender:
        text = " ".join(raw_text for raw_text in df.raw_text)
    else:
        text = " ".join(raw_text for raw_text in df[df["sender"] == sender].raw_text)

    # Create stopword list:
    stopwords = set(STOPWORDS)
    stopwords.update(["image", "omitted", "audio", "voice", "call",
                      "will", "know", "now", "ll", "re", "don"])

    if mask_path:
        # Generate mask
        mask = np.array(Image.open(mask_path))
        wordcloud = WordCloud(max_words=1000, background_color="white",
                              mode="RGBA", mask=mask, stopwords=stopwords)
        # create coloring from image
        image_colors = ImageColorGenerator(mask)
        wordcloud.generate(text)
        # Display wordcloud
        plt.figure(figsize=[10, 10])
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
    else:
        wordcloud = WordCloud(max_words=2000, background_color="white",
                              mode="RGBA", stopwords=stopwords)
        wordcloud.generate(text)
        # Display wordcloud
        plt.figure(figsize=[10, 10])
        plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    # store to file
    if write_path:
        plt.savefig(write_path, format="png")
        logger.info("Saving to {}".format(write_path))
    else:
        i = 0
        while os.path.exists("images/wordclouds/chat{}.png".format(i)):
            i += 1
        logger.info("Saving to images/wordclouds/chat{}.png".format(i))
        plt.savefig("images/wordclouds/chat{}.png".format(i), format="png")

    plt.draw()

def plot_message_count(df, time_frame='1D', trendline=False, write_path=None):
    """Plot message count for each user for various timeframes.

    df: Required. Dataframe containing chat information.
    time_frame: Can be used to get weekly message count: '7D', or monthly count '1M'.
                Default is daily count '1D'.
    trendline: Whether or not to include a trendline. Default is False.
    write_path: Supply path for plot to be saved to. Default is 'wordcloud/images/msgs_day{int}.png'.
    """
    # Setup colors and fonts
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rc('font', family='Comic Sans MS')
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
    df = df.set_index('timestamp')
    senders = {sender: df[df.sender == sender] for sender in df.sender.unique()}
    
    # Resample to given timeframe by summing
    for sender in senders:
        senders[sender] = senders[sender].resample(time_frame).count().reset_index()

    # Create figure and subplot
    fig, ax = plt.subplots()
    
    # Plot sender lines
    color_index = 0
    for i, sender in enumerate(senders):
        ax.plot(senders[sender].timestamp, senders[sender].raw_text, linewidth=1.5, color=colors[i])
        color_index = i + 1

    # calculate the trendline
    if trendline:
        x = [x for x in senders[sender].timestamp.index]
        y = senders[sender].raw_text.values
        z = np.polyfit(x, y, 14)
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
        logger.info("Saving to {}".format(write_path))
        fig.savefig(write_path, format = "PNG", dpi = 100)
    else:
        i = 0
        while os.path.exists("images/mgs_day/messages_per_day{}.png".format(i)):
            i += 1
        logger.info("Saving to images/mgs_day/messages_per_day{}.png".format(i))
        fig.savefig('images/mgs_day/messages_per_day{}.png'.format(i), format = "PNG", dpi = 100)
    plt.draw()

def get_general_stats(df, print_stats=False):
    r"""Get and print general statistics from the df.

    - Total messages sent per user:                 'total_msgs'
    - Total words sent per user:                    'total_words'
    - Most active day(most messages sent) per user: 'most_active_day'
    - \# of mesages sent on most active day:         'mad_msg_count'
    - Most active week(most messages sent) per user:'most_active_week'
    - \# of messages sent during most active week:   'maw_msg_count'
    - Least active week per user:                   'least_active_week'
    - \# of messages sent during least active week:  'law_msg_count'
    - Average messages per day per user:            'mean_daily_sent'
    - Average messages per week per user:           'mean_weekly_sent'
    - Average message length(chars) per user:       'mean_msg_len_char'
    - Average message length(words) per user:       'mean_msg_len_word'
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
        stat_dict[sender]['total_msgs'] = len(df.loc[df.sender == sender])
    
    # Total words sent
    for sender, group in df.groupby('sender'):
        message_lengths = group['raw_text'].map(lambda x: len(re.findall(r'\w+', x)) )
        stat_dict[sender]['total_words'] = math.ceil(message_lengths.sum())

    # Most active day
    for sender, group in df.groupby('sender'):
            # Set index to timestamp, resample per day, get message count per day
            group = group.set_index('timestamp')
            group = group.resample('1D').count().reset_index()
            # Select row of maximum value of raw_text(now the count of messages per day)
            max_row = group.loc[group.raw_text.idxmax()]
            # Select date from row
            max_timestamp = max_row.timestamp.date()
            stat_dict[sender]['most_active_day'] = max_timestamp
            stat_dict[sender]['mad_msg_count'] = int(max_row.raw_text)

    # Most active week + count for that week
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('7D').count().reset_index()
        max_row = group.loc[group.raw_text.idxmax()]
        max_timestamp = max_row.timestamp.date()
        stat_dict[sender]['most_active_week'] = max_timestamp
        stat_dict[sender]['maw_msg_count'] = int(max_row.raw_text)

    # Least active week + count for that week
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('7D').count().reset_index()
        min_row = group.loc[group.raw_text.idxmin()]
        min_timestamp = min_row.timestamp.date()
        stat_dict[sender]['least_active_week'] = min_timestamp
        stat_dict[sender]['law_msg_count'] = int(min_row.raw_text)

    # Average messages per day
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('1D').count().reset_index()
        average_sent = group.raw_text.mean()
        stat_dict[sender]['mean_daily_sent'] = math.ceil(average_sent)
    
    # Average messages per week
    for sender, group in df.groupby('sender'):
        group = group.set_index('timestamp')
        group = group.resample('7D').count().reset_index()
        average_sent = group.raw_text.mean()
        stat_dict[sender]['mean_weekly_sent'] = math.ceil(average_sent)

    # Average message length(chars)
    for sender, group in df.groupby('sender'):
        message_lengths = group['raw_text'].map(lambda x: len(x))
        stat_dict[sender]['mean_msg_len_char'] = math.ceil(message_lengths.mean())

    # Average message length(words)
    for sender, group in df.groupby('sender'):
        message_lengths = group['raw_text'].map(lambda x: len(re.findall(r'\w+', x)) )
        stat_dict[sender]['mean_msg_len_word'] = math.ceil(message_lengths.mean())

    if print_stats:
        for sender in stat_dict:
            stat_dict[sender]['most_active_day'] = stat_dict[sender]['most_active_day'].strftime('%a %d, %b %Y')
            # Format most_active_week
            start_date = stat_dict[sender]['most_active_week']
            end_date = start_date + timedelta(days = 7)
            start_date = start_date.strftime('%a %d, %b %Y')
            end_date = end_date.strftime('%a %d, %b %Y')
            stat_dict[sender]['most_active_week'] = start_date + " until " + end_date
            # Format least_active_week
            start_date = stat_dict[sender]['least_active_week']
            end_date = start_date + timedelta(days = 7)
            start_date = start_date.strftime('%a %d, %b %Y')
            end_date = end_date.strftime('%a %d, %b %Y')
            stat_dict[sender]['least_active_week'] = start_date + " until " + end_date
        logger.info("{}".format(json.dumps(stat_dict, indent = 2)))
        print(json.dumps(stat_dict, indent = 2))
    else:
        logger.info("{}".format(stat_dict))
    return stat_dict

def plot_msg_len_distrib(df):
    """Plot the distribution of message lengths for each sender in df."""
    pass

def plot_active_hour(df, write_path=None):
    """Plot the most active hour for each sender in df.
    
    The aggregate number of texts sent for each hour is calculated,
    this is then used to indicate which was the most popular hour to send messages.
    """
    # Calculate active_hours
    active_hours = {}
    for sender, group in df.groupby('sender'):
        group['Hour'] = group.apply(lambda row: row.timestamp.hour, axis = 1)
        text_counts = dict(group.Hour.value_counts())
        for hour in range(0,24):
            if hour not in text_counts:
                text_counts[hour] = 0
        active_hours[sender] = text_counts
    for sender in active_hours:
        xaxis = []
        xaxis.extend(active_hours[sender].values())
        active_hours[sender]['xaxis'] = xaxis
    yaxis = [i for i in range(1,25)]

    # Plot active_hours
    fig, ax = plt.subplots()
    # Color scheme
    colors = ['#26547c', '#ef476f', '#ffd166', '#06d6a0', '#98c1d9', '#ff99c9', '#3c1518']
    plt.rc('font', family='Arial')
    index = 0
    senders = active_hours.keys()
    for sender in senders:
        # color = plt.rcParams['axes.prop_cycle'].by_key()['color'][index+2]
        ax.scatter(active_hours[sender]['xaxis'], yaxis, c=colors[index], label=sender,
                    edgecolors='none')
        index += 1
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.set_ylabel('Hour of the day', fontsize = 12)
    ax.set_xlabel('Overall messages sent', fontsize = 12)
    ax.legend()
    plt.yticks(yaxis)

    # store to file
    if write_path:
        plt.savefig(write_path, format="png")
        logger.info("Saving to {}".format(write_path))
    else:
        i = 0
        while os.path.exists("images/active_hours/active_hours{}.png".format(i)):
            i += 1
        logger.info("Saving to images/active_hours/active_hours{}.png".format(i))
        plt.savefig("images/active_hours/active_hours{}.png".format(i), format="png")

    plt.draw()

def plot_general_stats(df, write_path=None):
    """Plot the statistics returned from get_general_stats()."""
    stats = get_general_stats(df)
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(18, 10)
    axs[0,0].bar(list(stats.keys()), [stats[sender]['total_msgs'] for sender in stats])
    axs[0,0].set_title("Total messages sent")
    axs[0,1].bar(list(stats.keys()), [stats[sender]['total_words'] for sender in stats])
    axs[0,1].set_title("Total words sent")
    axs[0,2].bar(list(stats.keys()), [stats[sender]['mad_msg_count'] for sender in stats])
    axs[0,2].set_title("Most messages sent in a day")
    axs[1,0].bar(list(stats.keys()), [stats[sender]['mean_weekly_sent'] for sender in stats])
    axs[1,0].set_title("Average number of messages/week")
    axs[1,1].bar(list(stats.keys()), [stats[sender]['mean_msg_len_word'] for sender in stats])
    axs[1,1].set_title("Average message length")
    axs[1,2].bar(list(stats.keys()), [stats[sender]['mean_daily_sent'] for sender in stats])
    axs[1,2].set_title("Average messages sent per day")

    for ax_arr in axs:
        for ax in ax_arr:
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("General statistics")

    # Optionally save plot
    if write_path:
        logger.info("Saving to {}".format(write_path))
        fig.savefig(write_path, format = "PNG", dpi = 100)
    else:
        i = 0
        while os.path.exists("images/general_stats/stats{}.png".format(i)):
            i += 1
        logger.info("Saving to images/general_stats/stats{}.png".format(i))
        fig.savefig('images/general_stats/stats{}.png'.format(i), format = "PNG", dpi = 100)
    plt.draw()

def analyse_sentiment(df, write_path=None):
    """Plot the overall sentiment of each sender.

    The sentiment is calculated for each message sent using the NLTK library.
    This gives a probability of the message being positive, neutral or negative 
    and the highest probability is taken as the sentiment for that message.
    The number of positive, neutral and negative messages is then aggregated and used
    to plot a pie chart. 
    """
    nltk.download('vader_lexicon')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    neutral, negative, positive = 0, 0, 0

    num_charts = len(df.sender.unique())
    fig, axs = plt.subplots(1, num_charts)
    fig.set_size_inches(11, 5)
    chart_index = 0

    for sender, group in df.groupby('sender'):
        all_text = group.raw_text
        l = len(all_text)
        # Initial call to print 0% progress
        hf.print_progress_bar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for index, sentence in enumerate(all_text):
            # Update Progress Bar
            hf.print_progress_bar(index + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
           
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

        logger.info("Negative: {0}% | Neutral: {1}% | Positive: {2}%".format(
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

    # Optionally save plot
    if write_path:
        logger.info("Saving to {}".format(write_path))
        fig.savefig(write_path, format = "PNG", dpi = 100)
    else:
        i = 0
        while os.path.exists("images/sentiments/sentiment{}.png".format(i)):
            i += 1
        logger.info("Saving to images/sentiments/sentiment{}.png".format(i))
        fig.savefig('images/sentiments/sentiment{}.png'.format(i), format = "PNG", dpi = 100)

    plt.draw()
