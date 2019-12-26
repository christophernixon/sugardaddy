#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import argparse
import os
import re
from datetime import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            print("Processed line {}".format(id))
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
        date = parse(raw_date)
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
    generate_wordcloud(df,sender=args.sender,mask_path=args.mask,write_path=args.dest)
