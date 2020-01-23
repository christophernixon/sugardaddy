import argparse

import matplotlib.pyplot as plt

import importer
import plot_maker as plots

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

    whatsapp = importer.Importer(args.filepath)
    plots.generate_wordcloud(whatsapp.df,sender = args.sender, mask_path = args.mask, write_path = args.dest)
    plots.plot_message_count(whatsapp.df, '1D', trendline = False)
    plots.analyse_sentiment(whatsapp.df)
    plots.plot_general_stats(whatsapp.df)
    plots.plot_active_hour(whatsapp.df)
    plt.show()
