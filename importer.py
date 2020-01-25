"""Imports and extracts whatsapp chat data."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import logging

import numpy as np
import pandas as pd
from dateutil.parser import parse

import helper_functions as hf


"""
Whatsapp line format:
"[dd/mm/yyyy, hh/mm/ss] SENDER: MESSAGE\n"
"""


class Importer:
    """Import a whatsapp format file and convert to a dataframe."""

    df = None

    def __init__(self, filename):
        """Parse whatsapp format file into dataframe."""
        self.logger = self.setup_logging()
        if os.path.exists(filename) and os.path.isfile(filename):
            self.df = self._parse_texts(filename)
        else:
            self.logger.error("Invalid filepath")

    def _file_len(self, fname):
        """Get # lines in a file."""
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def _parse_texts(self, raw_text_file):
        """Parse whatsapp format file into dataframe.

        Given a filepath to a .txt file with Whatsapp formatted lines,
        this returns a dataframe representing this file
        """
        index = 0
        text_table = []
        file_len = self._file_len(raw_text_file)
        self.logger.info("Importing {}".format(raw_text_file))
        with open(raw_text_file, 'r') as file:
            # Initial call to print 0% progress
            hf.print_progress_bar(0, file_len, prefix='Progress:', suffix='Complete', length=50)
            for line in file:
                index += 1
                hf.print_progress_bar(index, file_len, prefix='Progress:', suffix='Complete', length=50)
                tmp_array = []

                date = self._extract_date(line)
                sender = self._extract_sender(line)
                length = self._extract_length(line)
                message_body = self._extract_message(line)
                
                # TODO: Handle multi-line texts
                if date == "" or sender == "" or length == 0:
                    continue

                tmp_array.extend((index, date, sender, length, message_body))
                text_table.append(tmp_array)

                percent = round(100 * (index / float(file_len)),2)
                if percent % 5 == 0:
                    # Log approximately every 5% to get an estimate of which lines are processed.
                    self.logger.debug("Processed line %s", index)

        data = np.array(text_table)
        dataframe = pd.DataFrame({'text_id': data[:, 0], 'timestamp': data[:, 1],
                                  'sender': data[:, 2], 'length': data[:, 3],
                                  'raw_text': data[:, 4]})
        # Remove users who have only sent one message
        dataframe = dataframe.groupby('sender').filter(lambda x: x['raw_text'].size > 1.)
        return dataframe

    def _extract_date(self, line):
        """Extract date from whatsapp format string.
        
        Given a string in Whatsapp format,
        returns a datetime object of when the message was sent.
        """
        pattern = re.compile("(\[[^]]{20}\])")
        match = re.search(pattern, line)
        if match:
            raw_date = match.group()[1:-1]
            date = parse(raw_date, dayfirst=True)
        else:
            # TODO: Handle multi-line texts
            date = ""
        return date

    def _extract_sender(self, line):
        """Extract sender from whatsapp format string.

        Given a string in Whatsapp format,
        returns the sender of the message.
        """
        pattern = re.compile("(] [^:]*:)")
        match = re.search(pattern, line)
        if match:
            sender = match.group()[2:-1]
            if hf.is_dancer(sender):
                sender = "Clara"
        else:
            # TODO: Handle multi-line texts
            sender = ""
        return sender

    def _extract_length(self, line):
        """Extract sender from whatsapp format string.

        Given a string in Whatsapp format,
        returns the length of the message body of the string
        """
        pattern = re.compile("(] [^:]*: )")
        match = re.search(pattern, line)
        if match:
            sender_index = match.end()
            return len(line[sender_index:])
        else:
            # TODO: Handle multi-line texts
            return len(line)

    def _extract_message(self, line):
        """Extract message from whatsapp format string.

        Given a string in Whatsapp format,
        returns the message body of the string.
        """
        pattern = re.compile("(] [^:]*: )")
        match = re.search(pattern, line)
        if match:
            sender_index = match.end()
            return line[sender_index:].rstrip()
        else:
            # TODO: Handle multi-line texts
            return line

    def setup_logging(self):
        """Setup logging to file and console."""
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
        return logger
